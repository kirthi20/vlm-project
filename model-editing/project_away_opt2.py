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
    
    def __init__(self, device: torch.device =None, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
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

        # Optimal parameters for SmolVLM (adjusted for architecture)
        self.optimal_params = {
            'edit_layer': 8,  # Middle layers work best for SmolVLM
            'text_layer': 8,  # Match edit layer for consistency
            'weight': 0.8,    # Conservative weight to avoid over-correction
            'threshold': 0.15 # Confidence threshold
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
                # Return features before connector (for editing)
                return vision_features.view(batch_size, -1, vision_features.shape[-1])
            
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
            # Get average attention to image tokens from object tokens
            # This is complex for multi-modal models, so we'll use a simplified version
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
    
    def project_away(
        self,
        vision_features: torch.Tensor,
        objects_to_remove: List[str],
        weight: float = 1.0,
        edit_layer: int = 8,
        text_layer: int = 8
    ) -> torch.Tensor:
        """Apply ProjectAway algorithm to remove objects from vision representations."""
        # Work with pre-connector features
        edited_features = vision_features.clone()
        
        # Get text embeddings for objects to remove
        for obj in objects_to_remove:
            # Get text representation at specified layer
            text_direction = self._get_text_direction(obj, text_layer)
            
            # Project text direction to vision space if needed
            if text_direction.shape[-1] != edited_features.shape[-1]:
                # Use connector's inverse or a learned projection
                # For now, we'll use a simple linear projection
                text_direction = self._project_to_vision_space(text_direction)
            
            # Normalize direction
            text_direction = F.normalize(text_direction, dim=-1)
            
            # Apply ProjectAway to each patch
            for i in range(edited_features.shape[1]):
                patch_feature = edited_features[0, i]
                
                # Calculate projection
                projection = torch.dot(patch_feature, text_direction)
                
                # Remove component in text direction (only if positive)
                if projection > 0:
                    edited_features[0, i] = patch_feature - weight * projection * text_direction
        
        return edited_features
    
    def _get_text_direction(self, text: str, layer: int) -> torch.Tensor:
        """Get text direction vector at specified layer."""
        cache_key = f"{text}_{layer}"
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]
        
        # Create a contrastive prompt
        positive_prompt = f"This is a {text}."
        negative_prompt = "This is an image."
        
        # Tokenize both prompts
        pos_tokens = self.processor.tokenizer(
            positive_prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        neg_tokens = self.processor.tokenizer(
            negative_prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Get embeddings at specified layer
            pos_outputs = self.language_model(
                **pos_tokens,
                output_hidden_states=True
            )
            neg_outputs = self.language_model(
                **neg_tokens,
                output_hidden_states=True
            )
            
            # Get hidden states at specified layer
            pos_hidden = pos_outputs.hidden_states[layer].mean(dim=1).squeeze(0)
            neg_hidden = neg_outputs.hidden_states[layer].mean(dim=1).squeeze(0)
            
            # Direction is difference between positive and negative
            direction = pos_hidden - neg_hidden
        
        self.text_embedding_cache[cache_key] = direction
        return direction
    
    def _project_to_vision_space(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Project text embedding to vision feature space."""
        # If dimensions don't match, we need to project
        if text_embedding.shape[-1] == self.hidden_dim and self.vision_hidden_dim != self.hidden_dim:
            # Use the connector in reverse (approximate)
            # This is a simplification - ideally we'd have a learned projection
            with torch.no_grad():
                # Create a dummy tensor with right shape
                dummy = torch.zeros(1, 1, self.vision_hidden_dim, device=self.device, dtype=text_embedding.dtype)
                # Pass through connector to get shape
                connected = self.connector(dummy)
                
                # Simple linear projection (this could be improved)
                scale = self.vision_hidden_dim / self.hidden_dim
                projected = F.interpolate(
                    text_embedding.unsqueeze(0).unsqueeze(0),
                    size=self.vision_hidden_dim,
                    mode='linear'
                ).squeeze()
                
                return projected
        
        return text_embedding
    
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
                'object_confidences': {},  # Add this line
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
            
            # Ensure correct shape
            if edited_features.ndim == 4:
                batch_size = edited_features.shape[0]
                edited_features = edited_features.view(batch_size, -1, edited_features.shape[-1])
            
            # Temporarily replace the forward method
            original_forward = self.model.model.vision_model.forward
            original_connector = self.connector.forward
            
            def patched_vision_forward(pixel_values, **kwargs):
                # Return dummy output with correct structure
                outputs = type('obj', (object,), {
                    'last_hidden_state': edited_vision_features,
                    'hidden_states': None,
                    'attentions': None
                })()
                return outputs
            
            def patched_connector_forward(x):
                if torch.equal(x, edited_vision_features):
                    return edited_features
                return original_connector(x)
            
            self.model.model.vision_model.forward = patched_vision_forward
            self.connector.forward = patched_connector_forward
            
            try:
                # Generate with edited features
                with torch.no_grad():
                    cleaned_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
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
                
            finally:
                # Restore original methods
                self.model.model.vision_model.forward = original_forward
                self.connector.forward = original_connector
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
        
        # Find best threshold (not too many, not too few hallucinations)
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
        
        # Find best layer (one that produces changes)
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
    
    # Find optimal parameters for this image
    print("\nFinding optimal parameters...")
    # param_analysis = pa.find_optimal_params(image)
    # print(f"Best parameters found: {param_analysis['best_params']}")
    
    # Test with optimal parameters
    print("\nRunning with optimal parameters...")
    results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image in detail.",
        confidence_threshold=0.9
    )
    
    print("\nResults:")
    print("Original caption:", results['original_caption'])
    print("Object confidences:", results['object_confidences'])
    print("Detected hallucinations:", results['hallucinations'])
    
    if results['removed']:
        print("Cleaned caption:", results['cleaned_caption'])
    else:
        print("No hallucinations detected.")
    
    # Test with different parameters for comparison
    print("\n\nTesting with aggressive parameters...")
    aggressive_results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image in detail.",
        confidence_threshold=0.1,
        removal_weight=1.2,
        edit_layer=6,
        text_layer=6
    )
    
    print("Hallucinations found:", aggressive_results['hallucinations'])
    if aggressive_results['removed']:
        print("Aggressive cleaned caption:", aggressive_results['cleaned_caption'])
    
    # Test with conservative parameters
    print("\n\nTesting with conservative parameters...")
    conservative_results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image in detail.",
        confidence_threshold=0.25,
        removal_weight=0.5,
        edit_layer=10,
        text_layer=10
    )
    
    print("Hallucinations found:", conservative_results['hallucinations'])
    if conservative_results['removed']:
        print("Conservative cleaned caption:", conservative_results['cleaned_caption'])