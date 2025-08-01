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
        """Get text direction vector at specified layer with improved contrastive approach."""
        cache_key = f"{text}_{layer}"
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]
        
        # Use multiple contrastive prompts for robustness
        object_prompts = [
            f"An image of a {text}",
            f"A photo showing a {text}",
        ]
        
        generic_prompts = [
            "An image",
            "A photo",
        ]
        
        object_embeddings = []
        generic_embeddings = []
        
        with torch.no_grad():
            # Get embeddings for object prompts
            for prompt in object_prompts:
                tokens = self.processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.language_model(
                    **tokens,
                    output_hidden_states=True
                )
                
                # Use mean pooling over sequence
                hidden = outputs.hidden_states[layer]
                pooled = hidden.mean(dim=1).squeeze(0)
                object_embeddings.append(pooled)
            
            # Get embeddings for generic prompts
            for prompt in generic_prompts:
                tokens = self.processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.language_model(
                    **tokens,
                    output_hidden_states=True
                )
                
                hidden = outputs.hidden_states[layer]
                pooled = hidden.mean(dim=1).squeeze(0)
                generic_embeddings.append(pooled)
        
        # Direction is mean difference
        obj_embed = torch.stack(object_embeddings).mean(dim=0)
        generic_embed = torch.stack(generic_embeddings).mean(dim=0)
        direction = obj_embed - generic_embed

        if hasattr(self.vision_model, 'dtype'):
            direction = direction.to(self.vision_model.dtype)
        elif self.vision_model.embeddings.patch_embedding.weight.dtype == torch.float16:
            direction = direction.to(torch.float16)
        
        self.text_embedding_cache[cache_key] = direction
        return direction
    
    def _project_to_vision_space(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Project text embedding to vision feature space."""
        # If dimensions match, no projection needed
        if text_embedding.shape[-1] == self.vision_hidden_dim:
            return text_embedding
        
        # Simple linear interpolation for dimension matching
        if text_embedding.shape[-1] == self.hidden_dim:
            # Create a simple projection matrix with correct dtype
            device = text_embedding.device
            dtype = text_embedding.dtype
            
            # Create projection matrix with same dtype as input
            projection_matrix = torch.randn(
                self.vision_hidden_dim, 
                self.hidden_dim, 
                device=device,
                dtype=dtype  # Match the dtype
            ) * 0.02
            
            # Apply projection
            projected = F.linear(
                text_embedding.unsqueeze(0),
                projection_matrix
            ).squeeze(0)
            
            return projected
        
        return text_embedding
    
    def project_away(
        self,
        vision_features: torch.Tensor,
        objects_to_remove: List[str],
        weight: float = 1.0,
        edit_layer: int = 8,
        text_layer: int = 8
    ) -> torch.Tensor:
        """Apply ProjectAway algorithm to remove objects from vision representations."""
        edited_features = vision_features.clone()
        
        for obj in objects_to_remove:
            # Get text representation at specified layer
            text_direction = self._get_text_direction(obj, text_layer).to(edited_features.dtype)
            
            # Project to vision space if needed
            if text_direction.shape[-1] != self.vision_hidden_dim:
                text_direction = self._project_to_vision_space(text_direction)
            
            # Normalize direction
            text_direction = F.normalize(text_direction, dim=-1)
            
            # Apply ProjectAway to each spatial position
            if edited_features.ndim == 3:  # [batch, seq, hidden]
                for i in range(edited_features.shape[1]):
                    patch = edited_features[0, i]
                    
                    # Calculate projection
                    projection = torch.dot(patch, text_direction)
                    
                    # Remove component in text direction (only if positive)
                    if projection > 0:
                        edited_features[0, i] = patch - weight * projection * text_direction
            
            elif edited_features.ndim == 4:  # [batch, height, width, hidden]
                batch_size, height, width, hidden = edited_features.shape
                edited_flat = edited_features.view(batch_size, -1, hidden)
                
                for i in range(edited_flat.shape[1]):
                    patch = edited_flat[0, i]
                    projection = torch.dot(patch, text_direction)
                    
                    if projection > 0:
                        edited_flat[0, i] = patch - weight * projection * text_direction
                
                edited_features = edited_flat.view(batch_size, height, width, hidden)
        
        return edited_features
    
    def generate_with_edited_features(self, inputs, edited_features):
        """Generate text using edited vision features."""
        # Create a modified forward function
        original_model_forward = self.model.forward
        
        def custom_forward(input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
            # If pixel values are provided, we need to replace them with our edited features
            if pixel_values is not None:
                # Get text embeddings
                text_embeds = self.model.model.text_model.embeddings(input_ids)
                
                # Combine vision and text embeddings
                inputs_embeds = torch.cat([edited_features, text_embeds], dim=1)
                
                # Create combined attention mask
                vision_attention_mask = torch.ones(
                    edited_features.shape[:2],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
                
                # Forward through the language model
                outputs = self.model.model.text_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=kwargs.get('output_hidden_states', False),
                    output_attentions=kwargs.get('output_attentions', False),
                    return_dict=True
                )
                
                # Apply language model head
                lm_logits = self.model.lm_head(outputs.last_hidden_state)
                
                # Return in expected format
                from transformers.modeling_outputs import CausalLMOutputWithPast
                return CausalLMOutputWithPast(
                    logits=lm_logits,
                    past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                    hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                    attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                )
            
            # Otherwise, use original forward
            return original_model_forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        
        # Temporarily replace forward method
        self.model.forward = custom_forward
        
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
            # Restore original forward
            self.model.forward = original_model_forward
    
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