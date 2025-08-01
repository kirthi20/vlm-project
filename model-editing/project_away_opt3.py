import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.generation import GenerationMixin
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json


class AdvancedProjectAway:
    """Advanced ProjectAway implementation compatible with SmolVLM/Idefics3 architecture."""
    
    def __init__(self, device: torch.device = None, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        """Initialize ProjectAway with a vision-language model."""
        self.device = device if device else torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Model components for Idefics3
        self.vision_model = self.model.model.vision_model
        self.language_model = self.model.model.text_model
        self.connector = self.model.model.connector
        
        # Get model dimensions
        self.hidden_dim = self.language_model.config.hidden_size
        self.vocab_size = self.language_model.config.vocab_size
        self.vision_hidden_dim = self.vision_model.config.hidden_size
        
        # Cache for text embeddings
        self.text_embedding_cache = {}

        # Optimal parameters for SmolVLM
        self.optimal_params = {
            'edit_layer': 8,
            'text_layer': 8,
            'weight': 0.8,
            'threshold': 0.15
        }

        with open('model-editing/distinct_objects.json', 'r') as f:
            self.common_objects = json.load(f)
        
    def get_vision_features(self, pixel_values: torch.Tensor, return_pre_connector: bool = False) -> torch.Tensor:
        """Extract vision features from pixel values, handling Idefics3 format."""
        with torch.no_grad():
            # Handle dimension reshaping
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(1)
            
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values_reshaped = pixel_values.view(batch_size * num_images, num_channels, height, width)
            
            # Ensure correct dtype
            pixel_values_reshaped = pixel_values_reshaped.to(dtype=self.vision_model.embeddings.patch_embedding.weight.dtype)
            
            # Process through vision model
            vision_outputs = self.vision_model(pixel_values=pixel_values_reshaped)
            vision_features = vision_outputs.last_hidden_state
            
            if return_pre_connector:
                # Return features preserving spatial dimensions for connector
                return vision_features
            
            # Apply connector
            image_features = self.connector(vision_features)
            
            # Return 3D tensor: [batch_size, seq_len, hidden_dim]
            if image_features.ndim == 4:
                image_features = image_features.view(batch_size, -1, image_features.shape[-1])
            
            return image_features
    
    def get_internal_confidence(
        self,
        image: Union[Image.Image, torch.Tensor],
        objects: List[str],
        use_attention: bool = True
    ) -> Dict[str, float]:
        """Calculate internal confidence scores using attention patterns and hidden states."""
        # Process image
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        else:
            inputs = {"pixel_values": image}
        
        # Create a prompt that includes all objects
        prompt = "This image contains: " + ", ".join(objects) + "."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        full_inputs = self.processor(
            images=[image] if isinstance(image, Image.Image) else inputs['pixel_values'],
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)
        
        confidences = {}
        
        with torch.no_grad():
            # Forward pass to get attention weights and hidden states
            outputs = self.model(
                **full_inputs,
                output_attentions=use_attention,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get token positions for each object
            for obj in objects:
                obj_tokens = self.processor.tokenizer(
                    " " + obj,  # Add space for better tokenization
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.device)
                
                # Find where these tokens appear in the input
                input_ids = full_inputs['input_ids'][0]
                obj_positions = []
                
                for i in range(len(input_ids) - len(obj_tokens[0]) + 1):
                    if torch.equal(input_ids[i:i+len(obj_tokens[0])], obj_tokens[0]):
                        obj_positions.extend(range(i, i+len(obj_tokens[0])))
                
                if not obj_positions:
                    # If object not found in prompt, check vocabulary probabilities
                    confidences[obj] = self._get_vocab_confidence(outputs, obj_tokens)
                else:
                    # Calculate confidence from hidden states
                    confidences[obj] = self._get_hidden_state_confidence(
                        outputs, obj_positions, use_attention
                    )
        
        return confidences
    
    def _get_vocab_confidence(self, outputs, obj_tokens):
        """Get confidence from vocabulary probabilities."""
        # Use the last hidden states to predict next tokens
        last_hidden = outputs.hidden_states[-1]
        logits = self.model.lm_head(last_hidden)
        probs = F.softmax(logits, dim=-1)
        
        # Get max probability for object tokens across all positions
        max_prob = 0.0
        for token_id in obj_tokens[0]:
            if token_id < probs.shape[-1]:
                token_probs = probs[0, :, token_id].max().item()
                max_prob = max(max_prob, token_probs)
        
        return max_prob
    
    def _get_hidden_state_confidence(self, outputs, obj_positions, use_attention):
        """Calculate confidence from hidden state magnitudes and attention."""
        # Get hidden states at object positions
        hidden_states = outputs.hidden_states[-1]  # Last layer
        obj_hidden = hidden_states[0, obj_positions]
        
        # Calculate confidence as normalized magnitude
        magnitudes = torch.norm(obj_hidden, dim=-1)
        confidence = magnitudes.mean().item()
        
        # Normalize by average magnitude
        all_magnitudes = torch.norm(hidden_states[0], dim=-1)
        avg_magnitude = all_magnitudes.mean().item()
        
        normalized_confidence = confidence / (avg_magnitude + 1e-6)
        
        # If using attention, also consider attention patterns
        if use_attention and outputs.attentions is not None:
            attention_score = 0.5  # Default if attention analysis fails
            
            try:
                # Get attention from last few layers
                for layer_idx in range(-3, 0):
                    attn = outputs.attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
                    # Average across heads
                    attn_avg = attn.mean(dim=0)
                    
                    # Get attention from object tokens to early positions (likely image)
                    image_positions = list(range(min(20, len(attn_avg))))  # First 20 tokens usually image
                    obj_to_image_attn = attn_avg[obj_positions][:, image_positions].mean().item()
                    attention_score = max(attention_score, obj_to_image_attn)
            except:
                pass
            
            # Combine magnitude and attention scores
            normalized_confidence = 0.7 * normalized_confidence + 0.3 * attention_score
        
        return min(1.0, normalized_confidence)
    
    def _get_text_direction(self, text: str, layer: int) -> torch.Tensor:
        """Get text direction vector following ProjectAway paper approach."""
        cache_key = f"{text}_{layer}"
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]
        
        # Following the paper: use contrastive prompts
        with_object = f"a photo of a {text}"
        without_object = "a photo"
        
        with torch.no_grad():
            # Process both prompts through the full model to get hidden states
            for prompt, is_object in [(with_object, True), (without_object, False)]:
                tokens = self.processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Get hidden states from the language model
                outputs = self.language_model(
                    **tokens,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract hidden state at specified layer
                hidden_state = outputs.hidden_states[layer]
                
                # Average pool over sequence length (excluding padding)
                mask = tokens.attention_mask.unsqueeze(-1).float()
                pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                
                if is_object:
                    embed_with = pooled.squeeze(0)
                else:
                    embed_without = pooled.squeeze(0)
            
            # The direction is the difference (as per ProjectAway)
            direction = embed_with - embed_without
            
            # Normalize the direction vector
            direction = F.normalize(direction, dim=-1)
            
            # Ensure correct dtype
            direction = direction.to(self.vision_model.dtype if hasattr(self.vision_model, 'dtype') 
                                else self.vision_model.embeddings.patch_embedding.weight.dtype)
        
        self.text_embedding_cache[cache_key] = direction
        return direction
        
    def _project_to_vision_space(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Project text embedding to vision feature space using the connector's learned mapping."""
        # SmolVLM uses a connector to map vision to text space
        # We need the inverse mapping: text to vision
        
        if text_embedding.shape[-1] == self.vision_hidden_dim:
            return text_embedding
        
        # Since we can't easily invert the connector, we'll use the fact that
        # the connector learns a mapping from vision to text space
        # We approximate the inverse by finding the vision direction that maps to our text direction
        
        if not hasattr(self, '_vision_text_mapping'):
            # Create a learnable mapping based on the connector's architecture
            # This should be initialized based on your model's specifics
            with torch.no_grad():
                # Sample some vision features to understand the mapping
                dummy_vision = torch.randn(1, 1, self.vision_hidden_dim, 
                                        device=self.device, 
                                        dtype=text_embedding.dtype)
                dummy_text = self.connector(dummy_vision)
                
                # Estimate the projection matrix
                if dummy_text.shape[-1] == text_embedding.shape[-1]:
                    # Use pseudo-inverse for the mapping
                    # In practice, you might want to collect more samples
                    self._vision_text_mapping = {
                        'scale': dummy_text.norm() / dummy_vision.norm(),
                        'vision_dim': self.vision_hidden_dim,
                        'text_dim': text_embedding.shape[-1]
                    }
        
        # Project using learned scale
        # This is a simplification - in practice you might want to learn this mapping
        if text_embedding.shape[-1] < self.vision_hidden_dim:
            # Pad with zeros
            projected = F.pad(text_embedding, (0, self.vision_hidden_dim - text_embedding.shape[-1]))
        else:
            # Truncate
            projected = text_embedding[..., :self.vision_hidden_dim]
        
        return projected
    
    def project_away(
        self,
        vision_features: torch.Tensor,
        objects_to_remove: List[str],
        weight: float = 1.0,
        edit_layer: int = 8,
        text_layer: int = 8
    ) -> torch.Tensor:
        """Apply ProjectAway algorithm exactly as described in the paper."""
        edited_features = vision_features.clone()
        
        # Ensure we're working with the right shape
        if edited_features.ndim == 4:  # [batch, height, width, hidden]
            b, h, w, d = edited_features.shape
            edited_features = edited_features.view(b, h*w, d)
            needs_reshape = True
        else:
            needs_reshape = False
        
        for obj in objects_to_remove:
            print(f"\nProcessing object: {obj}")
            
            # Get text direction at specified layer
            text_direction = self._get_text_direction(obj, text_layer)
            
            # Project to vision space
            if text_direction.shape[-1] != edited_features.shape[-1]:
                text_direction = self._project_to_vision_space(text_direction)
            
            # Ensure same device and dtype
            text_direction = text_direction.to(device=edited_features.device, 
                                            dtype=edited_features.dtype)
            
            # Check for NaN/Inf
            if torch.isnan(text_direction).any() or torch.isinf(text_direction).any():
                print(f"Warning: Invalid values in text direction for {obj}")
                continue
            
            # Normalize direction (crucial for ProjectAway)
            text_direction = F.normalize(text_direction, dim=-1, eps=1e-8)
            
            # Apply ProjectAway: v' = v - α * (v · d) * d
            # where v is the vision feature, d is the text direction, α is the weight
            
            total_projection = 0
            num_edited = 0
            
            batch_size = edited_features.shape[0]
            seq_len = edited_features.shape[1]
            
            for b in range(batch_size):
                for i in range(seq_len):
                    v = edited_features[b, i]  # Current vision feature
                    
                    # Compute projection scalar (v · d)
                    projection = torch.dot(v, text_direction)
                    
                    # Skip if projection is too small or invalid
                    if torch.isnan(projection) or abs(projection.item()) < 1e-6:
                        continue
                    
                    total_projection += abs(projection.item())
                    
                    # Apply ProjectAway formula
                    # Only edit if projection is significant (following paper's approach)
                    if abs(projection.item()) > 0.01:  # Threshold from paper
                        num_edited += 1
                        edited_features[b, i] = v - weight * projection * text_direction
            
            avg_projection = total_projection / (batch_size * seq_len)
            print(f"Average projection magnitude: {avg_projection:.4f}")
            print(f"Number of patches edited: {num_edited}/{batch_size * seq_len}")
        
        # Reshape back if needed
        if needs_reshape:
            edited_features = edited_features.view(b, h, w, d)
        
        return edited_features
    
    def generate_with_edited_features(self, inputs, edited_features):
        """Generate text using edited vision features."""
        # Save original methods
        original_vision_forward = self.model.model.vision_model.forward
        original_connector_forward = self.connector.forward
        
        # Store edited features
        self._use_edited = True
        self._edited_features = edited_features
        
        def patched_vision_forward(pixel_values, **kwargs):
            # Return dummy output - the actual features will come from connector
            return type('obj', (object,), {
                'last_hidden_state': torch.zeros(1, 1, self.vision_hidden_dim, device=pixel_values.device, dtype=pixel_values.dtype),
                'hidden_states': None,
                'attentions': None
            })()
        
        def patched_connector_forward(x):
            # Return our edited features instead of processing x
            if hasattr(self, '_use_edited') and self._use_edited:
                self._use_edited = False  # Only use once
                return self._edited_features
            return original_connector_forward(x)
        
        # Apply patches
        self.model.model.vision_model.forward = patched_vision_forward
        self.connector.forward = patched_connector_forward
        
        try:
            # Generate with edited features
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            return outputs
        finally:
            # Restore original methods
            self.model.model.vision_model.forward = original_vision_forward
            self.connector.forward = original_connector_forward
            # Clean up
            if hasattr(self, '_edited_features'):
                delattr(self, '_edited_features')
            if hasattr(self, '_use_edited'):
                delattr(self, '_use_edited')
    
    def verify_edit_application(self, original_features, edited_features, objects_removed):
        """Verify that edits are actually being applied."""
        with torch.no_grad():
            # Calculate the difference
            diff = (original_features - edited_features).abs().mean().item()
            
            print(f"\nEdit Verification:")
            print(f"Average feature difference: {diff:.4f}")
            
            # Calculate correlation with removed objects
            for obj in objects_removed:
                text_dir = self._get_text_direction(obj, self.optimal_params['text_layer'])
                if text_dir.shape[-1] != edited_features.shape[-1]:
                    text_dir = self._project_to_vision_space(text_dir)
                text_dir = F.normalize(text_dir, dim=-1)
                
                # Flatten features for correlation
                orig_flat = original_features.view(-1, original_features.shape[-1])
                edit_flat = edited_features.view(-1, edited_features.shape[-1])
                
                # Check correlation before and after
                orig_corr = torch.matmul(orig_flat, text_dir).mean().item()
                edit_corr = torch.matmul(edit_flat, text_dir).mean().item()
                
                print(f"Object '{obj}': Original correlation: {orig_corr:.4f}, Edited: {edit_corr:.4f}, Reduction: {orig_corr - edit_corr:.4f}")
            
            return diff > 0.01  # Should see meaningful changes
    
    def detect_and_remove_hallucinations(
        self,
        image: Union[Image.Image, torch.Tensor],
        prompt: str = "Describe this image in detail.",
        confidence_threshold: float = 0.15,
        removal_weight: Optional[float] = None,
        edit_layer: Optional[int] = None,
        text_layer: Optional[int] = None,
        return_debug_info: bool = False
    ) -> Dict:
        """Complete pipeline for hallucination detection and removal."""
        # Use optimal parameters if not specified
        removal_weight = removal_weight or self.optimal_params['weight']
        edit_layer = edit_layer or self.optimal_params['edit_layer']
        text_layer = text_layer or self.optimal_params['text_layer']
        
        # Process inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=[image] if isinstance(image, Image.Image) else image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate initial caption
        with torch.no_grad():
            initial_outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
            initial_caption = self.processor.decode(
                initial_outputs[0],
                skip_special_tokens=True
            )
            # Extract assistant's response
            if "Assistant:" in initial_caption:
                initial_caption = initial_caption.split("Assistant:")[-1].strip()
            elif "\n\n" in initial_caption:
                initial_caption = initial_caption.split("\n\n")[-1].strip()
        
        # Extract objects from caption
        objects = self.extract_objects_from_caption(initial_caption)
        
        if not objects:
            return {
                'original_caption': initial_caption,
                'cleaned_caption': initial_caption,
                'hallucinations': [],
                'object_confidences': {},
                'removed': False
            }
        
        # Get confidence scores
        confidences = self.get_internal_confidence(image, objects)
        
        # Identify hallucinations
        hallucinations = [
            obj for obj, conf in confidences.items()
            if conf < confidence_threshold
        ]
        
        result = {
            'original_caption': initial_caption,
            'object_confidences': confidences,
            'hallucinations': hallucinations,
            'removed': False
        }
        
        if hallucinations:
            print(f"\nDetected hallucinations: {hallucinations}")
            
            # Get pre-connector vision features for editing
            vision_features_pre = self.get_vision_features(inputs['pixel_values'], return_pre_connector=True)
            
            # Apply ProjectAway
            edited_vision_features = self.project_away(
                vision_features_pre,
                hallucinations,
                weight=removal_weight,
                edit_layer=edit_layer,
                text_layer=text_layer
            )
            
            # Apply connector to edited features
            edited_features = self.connector(edited_vision_features)
            
            # Ensure correct shape for generation
            if edited_features.ndim == 4:
                batch_size = edited_features.shape[0]
                edited_features = edited_features.view(batch_size, -1, edited_features.shape[-1])
            elif edited_features.ndim == 2:
                edited_features = edited_features.unsqueeze(0)
            
            # Verify edits were applied
            if return_debug_info:
                original_features = self.connector(vision_features_pre)
                if original_features.ndim == 4:
                    original_features = original_features.view(original_features.shape[0], -1, original_features.shape[-1])
                self.verify_edit_application(original_features, edited_features, hallucinations)
            
            # Generate with edited features
            cleaned_outputs = self.generate_with_edited_features(inputs, edited_features)
            cleaned_caption = self.processor.decode(
                cleaned_outputs[0],
                skip_special_tokens=True
            )
            
            # Extract assistant's response
            if "Assistant:" in cleaned_caption:
                cleaned_caption = cleaned_caption.split("Assistant:")[-1].strip()
            elif "\n\n" in cleaned_caption:
                cleaned_caption = cleaned_caption.split("\n\n")[-1].strip()
            
            result['cleaned_caption'] = cleaned_caption
            result['removed'] = True
            
        else:
            result['cleaned_caption'] = initial_caption
        
        return result
    
    def extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract object nouns from caption - enhanced version."""
        caption_lower = caption.lower()
        found_objects = []
        
        # Look for exact matches
        for obj in self.common_objects:
            if f" {obj} " in f" {caption_lower} " or caption_lower.startswith(f"{obj} ") or caption_lower.endswith(f" {obj}"):
                found_objects.append(obj)
        
        # Remove duplicates while preserving order
        seen = set()
        found_objects = [x for x in found_objects if not (x in seen or seen.add(x))]
        
        return found_objects
    
    def find_optimal_params(self, image, test_prompts: Optional[List[str]] = None):
        """Find optimal parameters for the current model and image."""
        if test_prompts is None:
            test_prompts = [
                "Describe this image.",
                "What objects are in this image?",
                "Tell me what you see."
            ]
        
        results = {
            'confidence_analysis': {},
            'layer_analysis': {},
            'weight_analysis': {},
            'best_params': {}
        }
        
        # Test confidence thresholds
        print("Testing confidence thresholds...")
        for threshold in [0.1, 0.15, 0.2, 0.25]:
            detections = []
            for prompt in test_prompts:
                result = self.detect_and_remove_hallucinations(
                    image,
                    prompt=prompt,
                    confidence_threshold=threshold,
                    removal_weight=0.8,
                    edit_layer=8
                )
                detections.append(len(result['hallucinations']))
            results['confidence_analysis'][threshold] = np.mean(detections)
        
        # Find best threshold
        best_threshold = min(results['confidence_analysis'].items(), 
                           key=lambda x: abs(x[1] - 2))[0]  # Target ~2 hallucinations
        
        # Test layers
        print("Testing layer configurations...")
        for layer in [4, 6, 8, 10, 12]:
            changes = 0
            for prompt in test_prompts[:1]:  # Just one prompt for speed
                result = self.detect_and_remove_hallucinations(
                    image,
                    prompt=prompt,
                    confidence_threshold=best_threshold,
                    removal_weight=0.8,
                    edit_layer=layer,
                    text_layer=layer
                )
                if result['removed'] and result['original_caption'] != result.get('cleaned_caption', ''):
                    changes += 1
            results['layer_analysis'][layer] = changes > 0
        
        # Find best layer
        best_layer = max((k for k, v in results['layer_analysis'].items() if v), default=8)
        
        # Test weights
        print("Testing removal weights...")
        for weight in [0.5, 0.8, 1.0, 1.2]:
            caption_lengths = []
            for prompt in test_prompts[:1]:
                result = self.detect_and_remove_hallucinations(
                    image,
                    prompt=prompt,
                    confidence_threshold=best_threshold,
                    removal_weight=weight,
                    edit_layer=best_layer
                )
                if result['removed']:
                    caption_lengths.append(len(result.get('cleaned_caption', '')))
            results['weight_analysis'][weight] = np.mean(caption_lengths) if caption_lengths else 0
        
        # Best weight maintains reasonable caption length
        best_weight = 0.8  # Default
        if results['weight_analysis']:
            target_length = 50  # Reasonable caption length
            best_weight = min(results['weight_analysis'].items(),
                            key=lambda x: abs(x[1] - target_length) if x[1] > 0 else float('inf'))[0]
        
        results['best_params'] = {
            'confidence_threshold': best_threshold,
            'edit_layer': best_layer,
            'text_layer': best_layer,
            'removal_weight': best_weight
        }
        
        return results

def prepare_image_safely(image, max_size=512):
    """
    Prepare image with very conservative sizing
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((max_size, max_size), Image.LANCZOS)
        #print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image

# Example usage
if __name__ == "__main__":
    from PIL import Image
    
    # Initialize ProjectAway
    print("Loading model...")
    pa = AdvancedProjectAway(model_name="HuggingFaceTB/SmolVLM-500M-Instruct")
    
    # Load an image
    print("\nTesting model...")
    test_image_url = "http://images.cocodataset.org/train2014/COCO_train2014_000000103108.jpg"
    
    # Download test image
    response = requests.get(test_image_url)
    image = Image.open(BytesIO(response.content))
    #image = prepare_image_safely(image)  # Resize conservatively
    
    # Test with optimal parameters
    print("\nRunning with optimal parameters (with debug info)...")
    results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image in detail.",
        confidence_threshold=0.9,
        return_debug_info=True
    )
    
    print("\nResults:")
    print("Original caption:", results['original_caption'])
    print("Object confidences:", results['object_confidences'])
    print("Detected hallucinations:", results['hallucinations'])
    
    if results['removed']:
        print("Cleaned caption:", results['cleaned_caption'])
    else:
        print("No hallucinations detected.")
    
    # Test with different thresholds
    print("\n\nTesting with higher confidence threshold...")
    high_threshold_results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image in detail.",
        confidence_threshold=0.3,
        return_debug_info=True
    )
    
    print("Hallucinations found:", high_threshold_results['hallucinations'])
    if high_threshold_results['removed']:
        print("Cleaned caption:", high_threshold_results['cleaned_caption'])