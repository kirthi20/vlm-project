import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.generation import GenerationMixin
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import requests
from io import BytesIO

class ProjectAwayGeneration(GenerationMixin):
    """Custom generation mixin that applies ProjectAway during generation."""
    
    def __init__(self, model, edited_embeddings, original_inputs):
        self.model = model
        self.edited_embeddings = edited_embeddings
        self.original_inputs = original_inputs
        
    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Override to use edited embeddings
        model_inputs = self.model.prepare_inputs_for_generation(*args, **kwargs)
        
        # Replace image embeddings with edited ones on first pass
        if hasattr(self, 'first_pass') and self.first_pass:
            model_inputs['inputs_embeds'] = self.edited_embeddings
            self.first_pass = False
            
        return model_inputs


class AdvancedProjectAway:
    """Advanced ProjectAway implementation with zero-shot segmentation support."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        """Initialize ProjectAway with a vision-language model."""
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_image_size={"longest_edge": 512}  # Use dictionary format
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Model components
        self.vision_encoder = self.model.model.vision_model.encoder
        self.language_model = self.model.model.text_model
        self.vision_projection = self.model.model.connector
        
        # Get model dimensions
        self.hidden_dim = self.language_model.config.hidden_size
        self.vocab_size = self.language_model.config.vocab_size
        
        # Optimal parameters from paper
        self.optimal_params = {
            'instructblip': {'edit_layer': 6, 'text_layer': 0, 'weight': 0.5},
            'llava': {'edit_layer': 15, 'text_layer': 15, 'weight': 1.0},
            'default': {'edit_layer': 10, 'text_layer': 10, 'weight': 0.8}
        }
        
    def get_spatial_resolution(self, image_shape: torch.Size) -> Tuple[int, int]:
        """Get spatial resolution of vision encoder output."""
        # This depends on the vision encoder architecture
        # For most ViT models with patch size 14 or 16
        patch_size = 14  # Adjust based on your model
        h, w = image_shape[-2] // patch_size, image_shape[-1] // patch_size
        return h, w
    
    def localize_object(
        self,
        image: Union[Image.Image, torch.Tensor],
        object_name: str,
        threshold: float = 0.3,
        visualize: bool = True
    ) -> Dict:
        """Localize an object in the image using internal confidence scores.
        
        Args:
            image: Input image
            object_name: Name of object to localize
            threshold: Confidence threshold for segmentation
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with segmentation mask and visualization
        """
        # Process image
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            pil_image = image
        else:
            inputs = {"pixel_values": image}
            pil_image = None
            
        with torch.no_grad():
            # Get vision features
            vision_features = self.vision_encoder(inputs['pixel_values'])
            image_embeddings = self.vision_projection(vision_features)
            
        # Get spatial dimensions
        batch_size, num_patches, hidden_dim = image_embeddings.shape
        h, w = self.get_spatial_resolution(inputs['pixel_values'].shape)
        
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
        for layer in range(0, num_layers, 3):  # Check every 3rd layer for efficiency
            # Apply logit lens
            probs = self.apply_logit_lens(image_embeddings, layer)
            
            # Get max probability for object tokens
            for token_id in obj_tokens[0]:
                token_probs = probs[0, :, token_id]
                confidence_map = torch.maximum(confidence_map, token_probs)
        
        # Reshape to 2D
        confidence_map_2d = confidence_map[1:].reshape(h, w)  # Skip CLS token if present
        
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
        
        # Visualize if requested
        # if visualize and pil_image:
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
        #     # Original image
        #     axes[0].imshow(pil_image)
        #     axes[0].set_title('Original Image')
        #     axes[0].axis('off')
            
        #     # Confidence heatmap
        #     im = axes[1].imshow(result['confidence_map'], cmap='hot')
        #     axes[1].set_title(f'Confidence Map: {object_name}')
        #     axes[1].axis('off')
        #     plt.colorbar(im, ax=axes[1])
            
        #     # Segmentation mask overlay
        #     axes[2].imshow(pil_image)
        #     axes[2].imshow(mask, alpha=0.5, cmap='Blues')
        #     axes[2].set_title(f'Segmentation: {object_name}')
        #     axes[2].axis('off')
            
        #     plt.tight_layout()
        #     result['visualization'] = fig
            
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
        """Complete pipeline for hallucination detection and removal.
        
        Args:
            image: Input image
            prompt: Generation prompt
            confidence_threshold: Threshold for hallucination detection
            removal_weight: ProjectAway weight (None for auto)
            edit_layer: Layer to edit (None for auto)
            text_layer: Text embedding layer (None for auto)
            return_debug_info: Whether to return debugging information
            
        Returns:
            Dictionary with original and cleaned captions
        """
        # Auto-select parameters based on model type
        if removal_weight is None or edit_layer is None or text_layer is None:
            params = self.optimal_params.get('default')
            removal_weight = removal_weight or params['weight']
            edit_layer = edit_layer or params['edit_layer']
            text_layer = text_layer or params['text_layer']
            
        # Process inputs
        inputs = self.processor(
            images=image,
            text=f"{self.processor.image_token}{prompt}",
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
            ).replace(prompt, "").strip()
            
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
            # Get image embeddings
            with torch.no_grad():
                # Use the model's built-in vision processing
                model_outputs = self.model.model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids']
                )
                # Extract image embeddings (typically the first N tokens correspond to image)
                image_tokens = inputs['input_ids'].shape[1] - 1  # Subtract text tokens
                image_embeddings = model_outputs.last_hidden_state[:, :image_tokens, :]
                
            # Apply ProjectAway
            edited_embeddings = self.project_away(
                image_embeddings,
                hallucinations,
                weight=removal_weight,
                edit_layer=edit_layer,
                text_layer=text_layer
            )
            
            # Generate with edited embeddings
            # Create new inputs with edited embeddings
            edited_inputs = {
                'inputs_embeds': torch.cat([
                    edited_embeddings,
                    self.language_model.get_input_embeddings()(inputs.input_ids[:, 1:])
                ], dim=1),
                'attention_mask': torch.ones(
                    edited_embeddings.shape[0],
                    edited_embeddings.shape[1] + inputs.input_ids.shape[1] - 1,
                    device=self.device
                )
            }
            
            # Generate new caption
            with torch.no_grad():
                cleaned_outputs = self.language_model.generate(
                    **edited_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                cleaned_caption = self.processor.decode(
                    cleaned_outputs[0],
                    skip_special_tokens=True
                )
                
            result['cleaned_caption'] = cleaned_caption
            result['removed'] = True
            
            if return_debug_info:
                result['debug'] = {
                    'edit_layer': edit_layer,
                    'text_layer': text_layer,
                    'removal_weight': removal_weight,
                    'num_patches_edited': edited_embeddings.shape[1]
                }
        else:
            result['cleaned_caption'] = initial_caption
            
        return result
    
    def extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract object nouns from caption.
        
        Simple implementation - for better results use spaCy or similar.
        """
        # Common objects that might appear in captions
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
    
    def apply_logit_lens(
        self,
        embeddings: torch.Tensor,
        layer: int = -1
    ) -> torch.Tensor:
        """Apply logit lens to get vocabulary probabilities."""
        with torch.no_grad():
            if layer > 0:
                # Pass through layers
                attention_mask = torch.ones(
                    embeddings.shape[:2],
                    device=self.device
                )
                
                # Create position IDs
                position_ids = torch.arange(
                    embeddings.shape[1],
                    device=self.device
                ).unsqueeze(0)
                
                # Forward through layers using the language model directly
                # Use model's forward method with layer limit
                outputs = self.language_model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[layer] if layer < len(outputs.hidden_states) else outputs.last_hidden_state
                    
                # Apply LM head
                logits = self.model.lm_head(hidden_states)
            else:
                # Direct projection
                logits = self.model.lm_head(embeddings)
                
            return F.softmax(logits, dim=-1)
        
    def get_internal_confidence(
        self,
        image: torch.Tensor,
        objects: List[str],
        layers_to_check: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Calculate internal confidence scores for objects in an image.
        
        Args:
            image: Input image tensor
            objects: List of object names to check
            layers_to_check: Which layers to examine (None for all)
            
        Returns:
            Dictionary mapping object names to confidence scores
        """

        # Create a dummy prompt to get the model to process the image
        dummy_prompt = "Describe this image."
        
        # Process inputs properly through the processor
        inputs = self.processor(
            images=image,
            text=f"{self.processor.image_token}{dummy_prompt}",
            return_tensors="pt"
        ).to(self.device)

        # Process image
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get vision features directly
            model_outputs = self.model.model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids']
            )
            # Extract image embeddings (typically the first N tokens correspond to image)
            image_tokens = inputs['input_ids'].shape[1] - 1  # Subtract text tokens
            image_embeddings = model_outputs.last_hidden_state[:, :image_tokens, :]

        if layers_to_check is None:
            # Check all layers
            num_layers = len(self.language_model.layers)
            layers_to_check = list(range(num_layers))
            
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
                # Get probabilities at this layer
                probs = self.apply_logit_lens(image_embeddings, layer)
                
                # Get max probability for any token of this object
                for token_id in obj_tokens[0]:
                    token_tensor = probs[:, :, token_id]
                    if token_tensor.numel() > 0:
                        token_probs = token_tensor.max().item()
                        max_conf = max(max_conf, token_probs)
                    #max_conf = max(max_conf, token_probs)
                    
            confidences[obj] = max_conf
            
        return confidences


#Example usage and utilities
# def visualize_hallucination_removal(image_path: str, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
#     """Demo function to show hallucination removal."""
    
#     # Load model
#     pa = AdvancedProjectAway(model_name)
    
#     #Load image
#     image = Image.open(image_path)
    
#     #Run detection and removal
#     results = pa.detect_and_remove_hallucinations(
#         image,
#         prompt="Describe this image.",
#         confidence_threshold=0.15,
#         return_debug_info=True
#     )
    
#     print("Original caption:", results['original_caption'])
#     print("Detected hallucinations:", results['hallucinations'])
#     print("Object confidences:", results['object_confidences'])
    
#     if results['removed']:
#         print("Cleaned caption:", results['cleaned_caption'])
        
#     return results


# def demo_zero_shot_segmentation(image_path: str, object_name: str):
#     """Demo function for zero-shot segmentation."""
    
#     # Load model
#     pa = AdvancedProjectAway()
    
#     # Load image
#     image = Image.open(image_path)
    
#     # Perform segmentation
#     seg_results = pa.localize_object(
#         image,
#         object_name,
#         threshold=0.3,
#         visualize=True
#     )
    
#     print(f"Max confidence for '{object_name}': {seg_results['max_confidence']:.3f}")
    
#     return seg_results

# Example usage
if __name__ == "__main__":
    from PIL import Image
    
    # Initialize ProjectAway
    pa = AdvancedProjectAway("HuggingFaceTB/SmolVLM-256M-Instruct")
    
    # Load an image
    print("\nTesting model...")
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    
    # Download test image
    response = requests.get(test_image_url)
    image = Image.open(BytesIO(response.content))
    
    #Generate caption with hallucination reduction
    results = pa.detect_and_remove_hallucinations(
        image,
        prompt="Describe this image.",
        confidence_threshold=0.15,
        return_debug_info=True
    )
    print("Original caption:", results['original_caption'])
    print("Detected hallucinations:", results['hallucinations'])
    print("Object confidences:", results['object_confidences'])
    
    if results['removed']:
        print("Cleaned caption:", results['cleaned_caption'])
    else:
        print("No hallucinations detected, no cleaning performed.")