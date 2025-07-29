"""
Visual and Textual Intervention (VTI) for SmolVLM
Implementation based on "Reducing Hallucinations in Vision-Language Models via Latent Space Steering"
Paper: https://arxiv.org/pdf/2410.15778

This implementation provides a minimal and efficient way to apply VTI to SmolVLM-256M-Instruct
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
import random
from sklearn.decomposition import PCA
from transformers.image_utils import load_image


class VTIWrapper(nn.Module):
    """Wrapper to add VTI intervention to model layers"""
    def __init__(self, layer, direction, alpha=1.0):
        super().__init__()
        self.layer = layer
        self.direction = direction
        self.alpha = alpha
        self.enabled = True
    
    def forward(self, hidden_states, *args, **kwargs):
        # Apply the original layer
        output = self.layer(hidden_states, *args, **kwargs)
        
        # Apply VTI intervention if enabled
        if self.enabled and self.direction is not None:
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = hidden_states + self.alpha * self.direction
                output = (hidden_states,) + output[1:]
            else:
                output = output + self.alpha * self.direction
        
        return output


def add_vti_hooks(model, layer_directions, alpha=1.0, target_layers=None):
    """
    Add VTI intervention hooks to model layers
    
    Args:
        model: The model to add interventions to
        layer_directions: Dict mapping layer indices to intervention directions
        alpha: Strength of intervention
        target_layers: Optional list of specific layer indices to target
    """
    hooks = []
    
    def make_hook(direction):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Add the intervention
            hidden_states = hidden_states + alpha * direction.to(hidden_states.device)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        return hook
    
    # Apply hooks to specified layers
    for layer_idx, direction in layer_directions.items():
        if target_layers is None or layer_idx in target_layers:
            layer = model.model.get_submodule(f"layers.{layer_idx}")
            hook = layer.register_forward_hook(make_hook(direction))
            hooks.append(hook)
    
    return hooks

def prepare_image_safely(image, max_size=224):
    """
    Prepare image with very conservative sizing
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    
    # Scale down to a safe size
    scale = min(max_size / width, max_size / height)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Make dimensions divisible by 8 (often helps with vision models)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Ensure minimum size
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        #print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image

def compute_visual_direction(
    model,
    processor,
    demo_data: List[Tuple[str, str, str]],
    mask_ratio: float = 0.99,
    num_masks: int = 50,
    rank: int = 1
) -> Dict[int, torch.Tensor]:
    """
    Compute visual intervention directions by analyzing feature stability
    
    Args:
        model: The vision-language model
        processor: The model processor
        demo_images: List of demonstration images
        mask_ratio: Ratio of patches to mask (default 0.99)
        num_masks: Number of random masks to average over
        rank: PCA rank (typically 1)
    
    Returns:
        Dictionary mapping layer indices to intervention directions
    """
    model.eval()
    all_layer_shifts = {i: [] for i in range(len(model.model.vision_model.encoder.layers) + 1)}
    model_dtype = next(model.model.vision_model.parameters()).dtype

    with torch.no_grad():
        for img_url, _, _ in demo_data:
            img = load_image(img_url)
            img = prepare_image_safely(img, max_size=512)  # Ensure safe size

            # Process original image
            inputs = processor(images=img, return_tensors="pt").to(device=model.device, dtype=model_dtype)
            
            # Get original features
            orig_outputs = model.model.vision_model(
                pixel_values=inputs.pixel_values.flatten(0, 1),  # Shape: [13, 3, 512, 512]
                patch_attention_mask=inputs.get('patch_attention_mask', None).flatten(0, 1) if 'patch_attention_mask' in inputs else None,
                output_hidden_states=True
            )
            orig_hidden_states = orig_outputs.hidden_states
            
            # Collect features from multiple masked versions
            masked_features = [[] for _ in range(len(orig_hidden_states))]

            for _ in range(num_masks):
                # Create random mask
                masked_img = apply_random_mask(img, mask_ratio)
                masked_inputs = processor(images=masked_img, return_tensors="pt").to(model.device)
                
                # Get masked features
                masked_outputs = model.model.vision_model(
                    pixel_values=masked_inputs.pixel_values.flatten(0, 1).to(model_dtype),
                    output_hidden_states=True
                )
                
                for layer_idx, hidden_state in enumerate(masked_outputs.hidden_states):
                    masked_features[layer_idx].append(hidden_state)
            
            # Compute average masked features and shifts
            for layer_idx in range(len(orig_hidden_states)):
                avg_masked = torch.stack(masked_features[layer_idx]).mean(dim=0)
                shift = avg_masked - orig_hidden_states[layer_idx]
                all_layer_shifts[layer_idx].append(shift.cpu())
    
    # Apply PCA to extract principal directions
    visual_directions = {}
    for layer_idx, shifts in all_layer_shifts.items():
        if len(shifts) > 0:
            # Stack all shifts
            shifts_matrix = torch.cat(shifts, dim=0)
            
            # Reshape for PCA
            n_samples, seq_len, hidden_dim = shifts_matrix.shape
            shifts_flat = shifts_matrix.reshape(-1, hidden_dim).float().cpu().numpy()
            
            # Apply PCA
            pca = PCA(n_components=rank)
            pca.fit(shifts_flat)
            
            # Get principal direction
            principal_direction = torch.tensor(pca.components_[0], dtype=torch.float32)
            visual_directions[layer_idx] = principal_direction
    
    return visual_directions


def compute_textual_direction(
    model,
    tokenizer,
    processor,
    demo_data: List[Tuple[str, str, str]],
    rank: int = 1
) -> Dict[int, torch.Tensor]:
    """
    Compute textual intervention directions from hallucinated/clean caption pairs
    
    Args:
        model: The vision-language model
        tokenizer: The model tokenizer
        processor: The model processor
        demo_pairs: List of (clean_caption, hallucinated_caption, image) tuples
        rank: PCA rank (typically 1)
    
    Returns:
        Dictionary mapping layer indices to intervention directions
    """
    model.eval()
    all_layer_shifts = {i: [] for i in range(len(model.model.text_model.layers))}
    
    with torch.no_grad():
        for img_url, clean_text, hallucinated_text in demo_data:

            img = load_image(img_url)
            image = prepare_image_safely(img, max_size=512)  # Ensure safe size

            # Process clean caption with image
            clean_inputs = processor(
                images=image,
                text=clean_text,
                return_tensors="pt"
            ).to(model.device)
            
            # Process hallucinated caption with same image
            hall_inputs = processor(
                images=image, 
                text=hallucinated_text,
                return_tensors="pt"
            ).to(model.device)
            
            # Get hidden states for both when processing the captions
            clean_outputs = model(**clean_inputs, output_hidden_states=True)
            hall_outputs = model(**hall_inputs, output_hidden_states=True)
            
            # Extract text model hidden states
            clean_hidden = clean_outputs.text_model_outputs.hidden_states
            hall_hidden = hall_outputs.text_model_outputs.hidden_states
            
            # Compute shift: clean - hallucinated (Equation 2 in paper)
            for layer_idx in range(len(clean_hidden)):
                # Use last token representation as mentioned in paper
                shift = clean_hidden[layer_idx][:, -1, :] - hall_hidden[layer_idx][:, -1, :]
                all_layer_shifts[layer_idx].append(shift.cpu())
    
    # Apply PCA to extract principal directions
    textual_directions = {}
    for layer_idx, shifts in all_layer_shifts.items():
        if len(shifts) > 0:
            # Stack all shifts
            shifts_matrix = torch.cat(shifts, dim=0)
            shifts_flat = shifts_matrix.numpy()
            
            # Apply PCA
            pca = PCA(n_components=rank)
            pca.fit(shifts_flat)
            
            # Get principal direction
            principal_direction = torch.tensor(pca.components_[0], dtype=torch.float32)
            textual_directions[layer_idx] = principal_direction
    
    return textual_directions


def apply_random_mask(image: Image.Image, mask_ratio: float = 0.99) -> Image.Image:
    """Apply random patch masking to an image"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create patch grid (16x16 patches as typical for ViT)
    patch_size = 16
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    n_patches = n_patches_h * n_patches_w
    
    # Random mask
    n_masked = int(n_patches * mask_ratio)
    masked_indices = random.sample(range(n_patches), n_masked)
    
    # Apply mask
    masked_img = img_array.copy()
    for idx in masked_indices:
        row = idx // n_patches_w
        col = idx % n_patches_w
        masked_img[
            row * patch_size:(row + 1) * patch_size,
            col * patch_size:(col + 1) * patch_size
        ] = 0
    
    return Image.fromarray(masked_img)


class VTI:
    """Main VTI class for easy application to SmolVLM"""
    
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.visual_directions = None
        self.textual_directions = None
        self.vision_hooks = []
        self.text_hooks = []
    
    def compute_directions(
        self,
        demo_data: List[Tuple[str, str, str]],
        mask_ratio: float = 0.99,
        num_masks: int = 50
    ):
        """Compute both visual and textual intervention directions"""
        print("Computing textual directions...")
        self.textual_directions = compute_textual_direction(
            self.model, self.tokenizer, self.processor, demo_data
        )

        print("Computing visual directions...")
        self.visual_directions = compute_visual_direction(
            self.model, self.processor, demo_data, mask_ratio, num_masks
        )
        
        print(f"Computed directions for {len(self.visual_directions)} vision layers "
              f"and {len(self.textual_directions)} text layers")
    
    def apply_interventions(self, alpha_vision: float = 0.9, alpha_text: float = 0.9):
        """Apply VTI interventions to the model"""
        # Remove existing hooks
        self.remove_interventions()
        
        # Apply vision interventions
        if self.visual_directions:
            self.vision_hooks = add_vti_hooks(
                self.model.model.vision_model.encoder,
                self.visual_directions,
                alpha=alpha_vision
            )
        
        # Apply text interventions
        if self.textual_directions:
            self.text_hooks = add_vti_hooks(
                self.model.model.text_model,
                self.textual_directions,
                alpha=alpha_text
            )
    
    def remove_interventions(self):
        """Remove all VTI intervention hooks"""
        for hook in self.vision_hooks:
            hook.remove()
        for hook in self.text_hooks:
            hook.remove()
        self.vision_hooks = []
        self.text_hooks = []
    
    def save_directions(self, path: str):
        """Save computed directions to disk"""
        torch.save({
            'visual_directions': self.visual_directions,
            'textual_directions': self.textual_directions
        }, path)
    
    def load_directions(self, path: str):
        """Load pre-computed directions from disk"""
        checkpoint = torch.load(path)
        self.visual_directions = checkpoint['visual_directions']
        self.textual_directions = checkpoint['textual_directions']