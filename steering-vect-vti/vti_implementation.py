"""
Corrected Visual and Textual Intervention (VTI) for SmolVLM
Implementation based on "Reducing Hallucinations in Vision-Language Models via Latent Space Steering"
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


class VTI:
    """Corrected VTI implementation for SmolVLM"""
    
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.directions = None
        self.hooks = []
        self.intervention_layers = None
        self._debug_model_architecture()
    
    def _debug_model_architecture(self):
        """Debug SmolVLM architecture to find correct layer paths"""
        print("=== Debugging SmolVLM Architecture ===")
        
        # SmolVLM architecture exploration
        if hasattr(self.model, 'model'):
            print("Found model.model")
            model_inner = self.model.model
            
            # Check for language model component
            if hasattr(model_inner, 'text_model'):
                print("Found model.model.text_model")
                lm = model_inner.text_model
                if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                    self.language_layers = lm.model.layers
                    print(f"Found language layers: {len(self.language_layers)} layers")
                    self.layer_path = "model.language_model.model.layers"
                elif hasattr(lm, 'layers'):
                    self.language_layers = lm.layers
                    print(f"Found language layers: {len(self.language_layers)} layers")
                    self.layer_path = "model.language_model.layers"
            
            # Check for decoder
            elif hasattr(model_inner, 'decoder'):
                print("Found model.model.decoder")
                decoder = model_inner.decoder
                if hasattr(decoder, 'layers'):
                    self.language_layers = decoder.layers
                    print(f"Found decoder layers: {len(self.language_layers)} layers")
                    self.layer_path = "model.decoder.layers"
            
            # Direct layers check
            elif hasattr(model_inner, 'layers'):
                self.language_layers = model_inner.layers
                print(f"Found direct layers: {len(self.language_layers)} layers")
                self.layer_path = "model.layers"
        
        if not hasattr(self, 'language_layers'):
            print("ERROR: Could not find language model layers!")
            print("Available attributes on model.model:")
            for attr in dir(self.model.model):
                if not attr.startswith('_'):
                    print(f"  {attr}")
            raise RuntimeError("Could not locate language model layers in SmolVLM")
        
        print(f"Will use layer path: {self.layer_path}")
        print("=== Architecture Debug Complete ===\n")
    
    def compute_directions(
        self,
        demo_data: List[Tuple[str, str, str]],
        mask_ratio: float = 0.99,
        num_masks: int = 50,
        target_layers: Optional[List[int]] = None
    ):
        """
        Compute intervention directions from demonstration data
        
        Args:
            demo_data: List of (image_url, clean_caption, hallucinated_caption) tuples
            mask_ratio: Ratio of patches to mask for visual stability
            num_masks: Number of mask perturbations
            target_layers: Specific layers to target (default: last 1/3 of layers)
        """
        print("Computing VTI directions...")
        
        # Determine which layers to intervene on
        n_layers = len(self.language_layers)
        print(f"Model has {n_layers} language layers")
        
        if target_layers is None:
            # Intervene on last third of layers (where multimodal fusion happens)
            self.intervention_layers = list(range(2 * n_layers // 3, n_layers))
        else:
            self.intervention_layers = target_layers
            
        print(f"Will intervene on layers: {self.intervention_layers}")
        
        # Collect activation differences
        all_shifts = {layer_idx: [] for layer_idx in self.intervention_layers}
        
        self.model.eval()
        with torch.no_grad():
            for i, (img_url, clean_caption, hall_caption) in enumerate(demo_data):
                print(f"Processing demo sample {i+1}/{len(demo_data)}")
                try:
                    # Load and prepare image
                    image = load_image(img_url)
                    image = self._prepare_image(image)
                    
                    # Get shifts from both visual masking and caption differences
                    print("  Computing visual shift...")
                    visual_shift = self._compute_visual_shift(image, mask_ratio, min(num_masks, 10))
                    print("  Computing textual shift...")
                    textual_shift = self._compute_textual_shift(image, clean_caption, hall_caption)
                    
                    # Combine shifts (visual and textual information is entangled in LLM layers)
                    for layer_idx in self.intervention_layers:
                        if layer_idx in visual_shift and layer_idx in textual_shift:
                            # Average visual and textual shifts
                            combined_shift = (visual_shift[layer_idx] + textual_shift[layer_idx]) / 2
                            all_shifts[layer_idx].append(combined_shift)
                        elif layer_idx in visual_shift:
                            all_shifts[layer_idx].append(visual_shift[layer_idx])
                        elif layer_idx in textual_shift:
                            all_shifts[layer_idx].append(textual_shift[layer_idx])
                            
                except Exception as e:
                    print(f"  Error processing sample {i+1}: {e}")
                    continue
        
        # Apply PCA to extract principal directions
        self.directions = {}
        for layer_idx, shifts in all_shifts.items():
            if len(shifts) > 0:
                print(f"Computing direction for layer {layer_idx} with {len(shifts)} samples")
                
                # Stack all shifts
                shifts_matrix = torch.cat(shifts, dim=0)
                
                # Reshape for PCA
                if len(shifts_matrix.shape) == 3:  # [batch, seq, hidden]
                    shifts_flat = shifts_matrix.reshape(-1, shifts_matrix.shape[-1])
                else:
                    shifts_flat = shifts_matrix
                    
                shifts_flat = shifts_flat.float().cpu().numpy()
                
                # Apply PCA
                pca = PCA(n_components=1)
                pca.fit(shifts_flat)
                
                # Get principal direction
                direction = torch.tensor(pca.components_[0], dtype=torch.float32)
                direction = direction / torch.norm(direction)
                
                self.directions[layer_idx] = direction
                print(f"  Direction shape: {direction.shape}, norm: {torch.norm(direction):.4f}")
                
        print(f"Computed directions for {len(self.directions)} layers")
    
    def _compute_visual_shift(self, image, mask_ratio, num_masks):
        """Compute shift caused by visual masking"""
        shifts = {}
        
        # Create clean input
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image briefly."}
            ]
        }]
        
        # Get original activations
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.model.device)
        
        # Hook to capture activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                # Handle tuple outputs (hidden_states, attentions, etc.)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Store a copy to avoid gradient issues
                activations[name] = hidden_states.detach().clone()
            return hook
        
        # Register hooks on the correct layers
        hooks = []
        for layer_idx in self.intervention_layers:
            layer = self.language_layers[layer_idx]
            hook = layer.register_forward_hook(hook_fn(f"layer_{layer_idx}"))
            hooks.append(hook)
        
        # Get original activations
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        
        orig_activations = {k: v.cpu() for k, v in activations.items()}
        
        # Get masked activations
        masked_activations_list = []
        for mask_i in range(num_masks):
            masked_image = self._apply_random_mask(image, mask_ratio)
            masked_inputs = self.processor(text=prompt, images=[masked_image], return_tensors="pt").to(self.model.device)
            
            activations.clear()
            with torch.no_grad():
                _ = self.model.generate(**masked_inputs, max_new_tokens=1, do_sample=False)
            
            masked_activations_list.append({k: v.cpu() for k, v in activations.items()})
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute average shift
        for layer_idx in self.intervention_layers:
            key = f"layer_{layer_idx}"
            if key in orig_activations and len(masked_activations_list) > 0:
                orig = orig_activations[key]
                
                # Average masked activations
                valid_masked = [ma[key] for ma in masked_activations_list if key in ma]
                if len(valid_masked) > 0:
                    masked_avg = torch.stack(valid_masked).mean(dim=0)
                    
                    # Compute shift - focus on sequence positions with visual information
                    if len(orig.shape) == 3:  # [batch, seq, hidden]
                        # Average over all sequence positions for now
                        shift = orig.mean(dim=1) - masked_avg.mean(dim=1)
                    else:
                        shift = orig - masked_avg
                        
                    shifts[layer_idx] = shift
        
        return shifts
    
    def _compute_textual_shift(self, image, clean_caption, hall_caption):
        """Compute shift between clean and hallucinated captions"""
        shifts = {}
        
        # Create inputs for both captions
        clean_messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Caption: {clean_caption}"}
            ]
        }]
        
        hall_messages = [{
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Caption: {hall_caption}"}
            ]
        }]
        
        clean_prompt = self.processor.apply_chat_template(clean_messages, add_generation_prompt=False)
        hall_prompt = self.processor.apply_chat_template(hall_messages, add_generation_prompt=False)
        
        clean_inputs = self.processor(text=clean_prompt, images=[image], return_tensors="pt").to(self.model.device)
        hall_inputs = self.processor(text=hall_prompt, images=[image], return_tensors="pt").to(self.model.device)
        
        # Get hidden states using forward pass
        with torch.no_grad():
            # Forward pass to get all hidden states
            clean_outputs = self.model(**clean_inputs, output_hidden_states=True)
            hall_outputs = self.model(**hall_inputs, output_hidden_states=True)
        
        # Extract shifts from hidden states
        if hasattr(clean_outputs, 'hidden_states') and clean_outputs.hidden_states is not None:
            for layer_idx in self.intervention_layers:
                # Note: hidden_states includes embeddings as layer 0, so add 1 to layer_idx
                hidden_layer_idx = layer_idx + 1
                if hidden_layer_idx < len(clean_outputs.hidden_states):
                    clean_hidden = clean_outputs.hidden_states[hidden_layer_idx]
                    hall_hidden = hall_outputs.hidden_states[hidden_layer_idx]
                    
                    # Compute shift - average over sequence dimension
                    if len(clean_hidden.shape) == 3:  # [batch, seq, hidden]
                        shift = clean_hidden.mean(dim=1) - hall_hidden.mean(dim=1)
                    else:
                        shift = clean_hidden - hall_hidden
                    
                    shifts[layer_idx] = shift.cpu()
        
        return shifts
    
    def apply_interventions(self, alpha: float = 0.5):
        """
        Apply VTI interventions to the model
        
        Args:
            alpha: Intervention strength (0.5-2.0 recommended)
        """
        self.remove_interventions()
        
        if self.directions is None:
            raise ValueError("Must compute directions before applying interventions")
        
        print(f"Applying interventions with alpha={alpha} to {len(self.directions)} layers")
        
        def make_hook(direction, alpha, layer_idx):
            def hook(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    other_outputs = output[1:]
                else:
                    hidden_states = output
                    other_outputs = ()
                
                # Apply intervention
                direction_device = direction.to(hidden_states.device, dtype=hidden_states.dtype)
                
                if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                    # Apply to all positions
                    intervention = alpha * direction_device.unsqueeze(0).unsqueeze(0)
                    hidden_states = hidden_states + intervention
                elif len(hidden_states.shape) == 2:  # [batch, hidden]
                    intervention = alpha * direction_device.unsqueeze(0)
                    hidden_states = hidden_states + intervention
                else:
                    # Fallback for other shapes
                    intervention = alpha * direction_device
                    hidden_states = hidden_states + intervention
                
                # Return in same format as input
                if other_outputs:
                    return (hidden_states,) + other_outputs
                else:
                    return hidden_states
            return hook
        
        # Apply hooks to target layers
        for layer_idx, direction in self.directions.items():
            layer = self.language_layers[layer_idx]
            hook = layer.register_forward_hook(make_hook(direction, alpha, layer_idx))
            self.hooks.append(hook)
        
        print(f"Applied {len(self.hooks)} intervention hooks")
    
    def remove_interventions(self):
        """Remove all intervention hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        if len(self.hooks) == 0 and hasattr(self, 'directions') and self.directions:
            print("Removed all intervention hooks")
    
    def _prepare_image(self, image, max_size=512):
        """Prepare image safely"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed while maintaining aspect ratio
        width, height = image.size
        scale = min(max_size / width, max_size / height, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Make divisible by patch size (typically 14 for vision transformers)
            patch_size = 14
            new_width = (new_width // patch_size) * patch_size
            new_height = (new_height // patch_size) * patch_size
            new_width = max(new_width, patch_size)
            new_height = max(new_height, patch_size)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    def _apply_random_mask(self, image, mask_ratio):
        """Apply random patch masking"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create patch grid (use 16x16 patches for better compatibility)
        patch_size = 16
        n_patches_h = h // patch_size
        n_patches_w = w // patch_size
        n_patches = n_patches_h * n_patches_w
        
        if n_patches == 0:
            return image  # Image too small to patch
        
        # Random mask
        n_masked = int(n_patches * mask_ratio)
        n_masked = min(n_masked, n_patches)  # Ensure we don't exceed available patches
        
        if n_masked > 0:
            masked_indices = random.sample(range(n_patches), n_masked)
            
            # Apply mask (set to gray) - create a copy first
            masked_img = img_array.copy()
            gray_value = 128  # Mid-gray
            
            for idx in masked_indices:
                row = idx // n_patches_w
                col = idx % n_patches_w
                y_start = row * patch_size
                y_end = min((row + 1) * patch_size, h)
                x_start = col * patch_size  
                x_end = min((col + 1) * patch_size, w)
                
                # Ensure we don't go out of bounds and handle shape correctly
                if y_end > y_start and x_end > x_start:
                    if len(masked_img.shape) == 3:  # RGB image
                        masked_img[y_start:y_end, x_start:x_end, :] = gray_value
                    else:  # Grayscale
                        masked_img[y_start:y_end, x_start:x_end] = gray_value
            
            return Image.fromarray(masked_img.astype(np.uint8))
        
        return image
    
    def save_directions(self, path):
        """Save computed directions"""
        torch.save({
            'directions': self.directions,
            'intervention_layers': self.intervention_layers,
            'layer_path': self.layer_path
        }, path)
        print(f"Saved directions to {path}")
    
    def load_directions(self, path):
        """Load pre-computed directions"""
        checkpoint = torch.load(path, map_location='cpu')
        self.directions = checkpoint['directions']
        self.intervention_layers = checkpoint.get('intervention_layers', list(self.directions.keys()))
        print(f"Loaded directions for {len(self.directions)} layers from {path}")