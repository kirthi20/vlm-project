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


class AdvancedProjectAway:
    """Advanced ProjectAway implementation compatible with SmolVLM/Idefics3 architecture."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        """Initialize ProjectAway with a vision-language model."""
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
        
        # Cache for text embeddings
        self.text_embedding_cache = {}

        # Optimal parameters from paper
        self.optimal_params = {
            'instructblip': {'edit_layer': 6, 'text_layer': 0, 'weight': 0.5},
            'llava': {'edit_layer': 15, 'text_layer': 15, 'weight': 1.0},
            'default': {'edit_layer': 10, 'text_layer': 10, 'weight': 0.8}
        }
        
    def get_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features from pixel values, handling Idefics3 format."""
        with torch.no_grad():
            # For Idefics3, pixel_values should be [batch_size, num_images, num_channels, height, width]
            # If it's [batch_size, num_channels, height, width], add the num_images dimension
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(1)  # Add num_images dimension
            
            # The vision model expects [batch_size * num_images, num_channels, height, width]
            # So we need to reshape it
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values_reshaped = pixel_values.view(batch_size * num_images, num_channels, height, width)
            
            # Ensure pixel values have the same dtype as the model
            pixel_values_reshaped = pixel_values_reshaped.to(dtype=self.vision_model.embeddings.patch_embedding.weight.dtype)
            
            # Process through vision model
            vision_outputs = self.vision_model(pixel_values=pixel_values_reshaped)
            
            # Get the last hidden states
            vision_features = vision_outputs.last_hidden_state
            
            # Apply connector to project to language model dimension
            image_features = self.connector(vision_features)
            
            # Always return 3D tensor: [batch_size, seq_len, hidden_dim]
            if image_features.ndim == 4:
                # Flatten spatial dimensions
                image_features = image_features.view(batch_size, -1, image_features.shape[-1])
            
            return image_features
    
    def localize_object(
        self,
        image: Union[Image.Image, torch.Tensor],
        object_name: str,
        threshold: float = 0.3,
        visualize: bool = True
    ) -> Dict:
        """Localize an object in the image using internal confidence scores."""
        # Process image
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            pil_image = image
        else:
            inputs = {"pixel_values": image}
            pil_image = None
            
        # Get vision features
        image_embeddings = self.get_vision_features(inputs['pixel_values'])
        
        # Get spatial dimensions
        batch_size, num_patches, hidden_dim = image_embeddings.shape
        # Estimate spatial dimensions (assuming square patches)
        h = w = int(np.sqrt(num_patches))
        
        # Calculate confidence for each patch
        confidence_map = torch.zeros(num_patches, device=self.device)
        
        # Get token IDs for object
        obj_tokens = self.processor.tokenizer(
            object_name,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)
        
        # Check multiple layers
        num_layers = min(24, len(self.language_model.layers))
        for layer in range(0, num_layers, 3):
            # Apply logit lens
            probs = self.apply_logit_lens(image_embeddings, layer)
            
            # Get max probability for object tokens
            for token_id in obj_tokens[0]:
                token_probs = probs[0, :, token_id]
                confidence_map = torch.maximum(confidence_map, token_probs)
        
        # Reshape to 2D
        confidence_map_2d = confidence_map.reshape(h, w)
        
        # Resize to original image size
        if pil_image:
            orig_h, orig_w = pil_image.height, pil_image.width
        else:
            orig_h, orig_w = inputs['pixel_values'].shape[-2:]
            
        confidence_map_resized = F.interpolate(
            confidence_map_2d.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Create binary mask
        mask = (confidence_map_resized > threshold).cpu().numpy()
        
        result = {
            'mask': mask,
            'confidence_map': confidence_map_resized.cpu().numpy(),
            'max_confidence': confidence_map.max().item()
        }
            
        return result
    
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
        # Auto-select parameters based on model type
        if removal_weight is None or edit_layer is None or text_layer is None:
            params = self.optimal_params.get('default')
            removal_weight = removal_weight or params['weight']
            edit_layer = edit_layer or params['edit_layer']
            text_layer = text_layer or params['text_layer']
            
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
                max_new_tokens=50,
                do_sample=False
            )
            initial_caption = self.processor.decode(
                initial_outputs[0],
                skip_special_tokens=True
            )
            # Extract just the assistant's response
            if "Assistant:" in initial_caption:
                initial_caption = initial_caption.split("Assistant:")[-1].strip()
            elif "\n\n" in initial_caption:
                initial_caption = initial_caption.split("\n\n")[-1].strip()
            
        # Extract nouns/objects from caption
        objects = self.extract_objects_from_caption(initial_caption)
        
        if not objects:
            return {
                'original_caption': initial_caption,
                'cleaned_caption': initial_caption,
                'hallucinations': [],
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
            # Get the edited features
            image_features = self.get_vision_features(inputs['pixel_values'])
            
            # Apply ProjectAway
            edited_features = self.project_away(
                image_features,
                hallucinations,
                weight=removal_weight,
                edit_layer=edit_layer,
                text_layer=text_layer
            )
            
            # Store the edited features
            self._edited_features = edited_features
            
            # Patch the connector
            original_connector_forward = self.connector.forward
            
            def patched_connector(x):
                if hasattr(self, '_edited_features') and x.shape == self.vision_model(inputs['pixel_values']).last_hidden_state.shape:
                    return self._edited_features
                return original_connector_forward(x)
            
            self.connector.forward = patched_connector
            
            try:
                with torch.no_grad():
                    cleaned_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False
                    )
                    cleaned_caption = self.processor.decode(
                        cleaned_outputs[0],
                        skip_special_tokens=True
                    )
                    # Extract just the assistant's response
                    if "Assistant:" in cleaned_caption:
                        cleaned_caption = cleaned_caption.split("Assistant:")[-1].strip()
                    elif "\n\n" in cleaned_caption:
                        cleaned_caption = cleaned_caption.split("\n\n")[-1].strip()
            finally:
                # Clean up
                self.connector.forward = original_connector_forward
                if hasattr(self, '_edited_features'):
                    delattr(self, '_edited_features')
            
            result['cleaned_caption'] = cleaned_caption
            result['removed'] = True
        else:
            result['cleaned_caption'] = initial_caption
            
        return result
    
    def extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract object nouns from caption."""
        common_objects = [
            'person', 'man', 'woman', 'child', 'dog', 'cat', 'car', 'truck',
            'bicycle', 'motorcycle', 'airplane', 'bus', 'train', 'boat',
            'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'ball', 'kite', 'bat', 'glove',
            'skateboard', 'surfboard', 'racket', 'bottle', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'plant', 'bed', 'table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'bear', 'doll', 'hair', 'brush'
        ]
        
        caption_lower = caption.lower()
        found_objects = []
        
        for obj in common_objects:
            if obj in caption_lower:
                found_objects.append(obj)
                
        return found_objects
    
    def apply_logit_lens(self, embeddings: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """Apply logit lens to get vocabulary probabilities."""
        with torch.no_grad():
            # Ensure embeddings are properly shaped
            if embeddings.ndim == 4:  # If we have extra dimensions, flatten them
                batch_size = embeddings.shape[0]
                embeddings = embeddings.view(batch_size, -1, embeddings.shape[-1])
            
            if layer > 0:
                # For SmolVLM/Idefics3, we need to use the language model properly
                # Create dummy input_ids to match the sequence length
                seq_len = embeddings.shape[1]
                dummy_input_ids = torch.zeros((embeddings.shape[0], seq_len), dtype=torch.long, device=self.device)
                
                # Create attention mask
                attention_mask = torch.ones_like(dummy_input_ids)
                
                # Pass through the language model layers
                outputs = self.language_model(
                    input_ids=dummy_input_ids,
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get hidden states at specified layer
                if layer < len(outputs.hidden_states):
                    hidden_states = outputs.hidden_states[layer]
                else:
                    hidden_states = outputs.last_hidden_state
            else:
                hidden_states = embeddings
                
            # Apply LM head to get logits
            logits = self.model.lm_head(hidden_states)
            
            # Verify we have the full vocabulary
            if logits.shape[-1] != self.vocab_size:
                print(f"Warning: Expected vocab size {self.vocab_size}, got {logits.shape[-1]}")
                
            return F.softmax(logits, dim=-1)
        
    def get_internal_confidence(
        self,
        image: Union[Image.Image, torch.Tensor],
        objects: List[str],
        layers_to_check: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Calculate internal confidence scores for objects in an image."""
        # Process image
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        else:
            inputs = {"pixel_values": image}
            
        # Get image embeddings
        image_embeddings = self.get_vision_features(inputs['pixel_values'])
        
        if layers_to_check is None:
            # Check every 3rd layer for efficiency
            num_layers = min(24, len(self.language_model.layers))
            layers_to_check = list(range(0, num_layers, 3))
            
        confidences = {}
        
        for obj in objects:
            # Tokenize object name
            obj_tokens = self.processor.tokenizer(
                obj,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            max_conf = 0.0
            
            # Check each layer
            for layer in layers_to_check:
                try:
                    # Get probabilities at this layer
                    probs = self.apply_logit_lens(image_embeddings, layer)
                    
                    # Get max probability for any token of this object
                    for token_id in obj_tokens[0]:
                        if token_id < probs.shape[-1]:  # Ensure token_id is valid
                            token_probs = probs[:, :, token_id].max().item()
                            max_conf = max(max_conf, token_probs)
                except Exception as e:
                    print(f"Warning: Failed to get confidence at layer {layer} for object '{obj}': {e}")
                    continue
                        
            confidences[obj] = max_conf
            
        return confidences

    def project_away(
        self,
        image_embeddings: torch.Tensor,
        objects_to_remove: List[str],
        weight: float = 1.0,
        edit_layer: int = 15,
        text_layer: int = 15
    ) -> torch.Tensor:
        """Apply ProjectAway algorithm to remove objects from image representations."""
        edited_embeddings = image_embeddings.clone()
        
        for obj in objects_to_remove:
            # Get text embedding for this object
            text_embedding = self.get_text_embedding(obj, text_layer)
            
            # Normalize text embedding
            text_embedding = F.normalize(text_embedding, dim=-1)
            
            # For each image patch embedding
            for i in range(edited_embeddings.shape[1]):
                patch_embedding = edited_embeddings[0, i]
                
                # Calculate dot product
                dot_product = torch.dot(patch_embedding, text_embedding)
                
                # Only subtract if dot product is positive
                if dot_product > 0:
                    # Subtract weighted text embedding
                    edited_embeddings[0, i] = patch_embedding - weight * dot_product * text_embedding
                    
        return edited_embeddings
    
    def get_text_embedding(self, text: str, layer: int = -1) -> torch.Tensor:
        """Get text embedding for a given object at specified layer."""
        cache_key = f"{text}_{layer}"
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]
            
        # Tokenize the text
        text_inputs = self.processor.tokenizer(
            text, 
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            if layer == -1:
                # Use the embedding layer directly
                text_embeddings = self.language_model.get_input_embeddings()(
                    text_inputs.input_ids
                )
                # Take the mean of token embeddings
                text_embedding = text_embeddings.mean(dim=1).squeeze(0)
            else:
                # Run through model to get intermediate layer representation
                outputs = self.language_model(
                    input_ids=text_inputs.input_ids,
                    output_hidden_states=True
                )
                # Get representation at specified layer
                hidden_states = outputs.hidden_states[layer]
                # Take mean across tokens
                text_embedding = hidden_states.mean(dim=1).squeeze(0)
                
        self.text_embedding_cache[cache_key] = text_embedding
        return text_embedding


# Example usage
if __name__ == "__main__":
    from PIL import Image
    
    # Initialize ProjectAway
    print("Loading model...")
    pa = AdvancedProjectAway("HuggingFaceTB/SmolVLM-256M-Instruct")
    
    # Load an image
    print("\nTesting model...")
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    
    # Download test image
    response = requests.get(test_image_url)
    image = Image.open(BytesIO(response.content))
    
    # Generate caption with hallucination reduction
    print("\nRunning hallucination detection and removal...")
    results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image.",
        confidence_threshold=0.15,
        return_debug_info=True
    )
    
    print("\nResults:")
    print("Original caption:", results['original_caption'])
    print("Detected hallucinations:", results['hallucinations'])
    print("Object confidences:", results['object_confidences'])
    
    if results['removed']:
        print("Cleaned caption:", results['cleaned_caption'])
    else:
        print("No hallucinations detected, no cleaning performed.")