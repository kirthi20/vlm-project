"""
Corrected Visual and Textual Intervention (VTI) for SmolVLM
Implementation based on "Reducing Hallucinations in Vision-Language Models via Latent Space Steering"
This version correctly implements separate interventions for vision encoder and text decoder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
import random
from sklearn.decomposition import PCA
from transformers.image_utils import load_image


class VTI:
    """Corrected VTI implementation for SmolVLM with separate vision and text interventions"""
    
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.visual_directions = None
        self.textual_directions = None
        self.vision_hooks = []
        self.text_hooks = []
        self.vision_layers = None
        self.text_intervention_layers = None
        
        # Identify model architecture
        self._identify_model_components()
    
    def _identify_model_components(self):
        """Identify vision encoder and text decoder components"""
        print("Identifying model components...")
        
        # Find vision encoder
        self.vision_encoder = None
        self.text_decoder = None
        
        # Common paths for vision encoder in VLMs
        vision_paths = [
            'model.vision_tower.vision_tower.vision_model',
            'model.vision_tower.vision_model',
            'model.vision_tower',
            'vision_model',
            'visual_encoder'
        ]
        
        for path in vision_paths:
            try:
                component = self.model
                for attr in path.split('.'):
                    component = getattr(component, attr)
                if hasattr(component, 'encoder') or hasattr(component, 'layers'):
                    self.vision_encoder = component
                    print(f"Found vision encoder at: {path}")
                    break
            except AttributeError:
                continue
        
        # Find text decoder
        text_paths = [
            'model.text_model',
            'model.language_model',
            'model',
            'text_decoder',
            'language_model'
        ]
        
        for path in text_paths:
            try:
                component = self.model
                for attr in path.split('.'):
                    component = getattr(component, attr)
                if hasattr(component, 'layers'):
                    self.text_decoder = component
                    print(f"Found text decoder at: {path}")
                    break
            except AttributeError:
                continue
        
        # Get layer counts
        if self.vision_encoder:
            if hasattr(self.vision_encoder, 'encoder') and hasattr(self.vision_encoder.encoder, 'layers'):
                self.vision_layers = list(range(len(self.vision_encoder.encoder.layers)))
            elif hasattr(self.vision_encoder, 'layers'):
                self.vision_layers = list(range(len(self.vision_encoder.layers)))
            print(f"Vision encoder has {len(self.vision_layers) if self.vision_layers else 0} layers")
        
        if self.text_decoder and hasattr(self.text_decoder, 'layers'):
            n_text_layers = len(self.text_decoder.layers)
            # Intervene on last 1/3 of text decoder layers
            self.text_intervention_layers = list(range(2 * n_text_layers // 3, n_text_layers))
            print(f"Text decoder has {n_text_layers} layers, will intervene on layers {self.text_intervention_layers}")
    
    def compute_directions(
        self,
        demo_data: List[Tuple[str, str, str]],
        mask_ratio: float = 0.99,
        num_masks: int = 50,
        target_layers: Optional[Dict[str, List[int]]] = None
    ):
        """
        Compute intervention directions from demonstration data
        
        Args:
            demo_data: List of (image_url, clean_caption, hallucinated_caption) tuples
            mask_ratio: Ratio of patches to mask for visual stability
            num_masks: Number of mask perturbations
            target_layers: Dict with 'vision' and 'text' layer indices (optional)
        """
        print("Computing VTI directions...")
        
        if target_layers:
            if 'vision' in target_layers:
                self.vision_layers = target_layers['vision']
            if 'text' in target_layers:
                self.text_intervention_layers = target_layers['text']
        
        # Collect activation differences for vision and text separately
        visual_shifts = {layer_idx: [] for layer_idx in (self.vision_layers or [])}
        textual_shifts = {layer_idx: [] for layer_idx in (self.text_intervention_layers or [])}
        
        self.model.eval()
        with torch.no_grad():
            for img_url, clean_caption, hall_caption in demo_data:
                #print(f"Processing example: {img_url[:50]}...")
                
                # Load and prepare image
                image = load_image(img_url)
                image = self._prepare_image(image)
                
                # Compute visual shifts (from masking perturbations)
                if self.vision_layers:
                    v_shifts = self._compute_visual_shift(image, mask_ratio, num_masks)
                    for layer_idx, shift in v_shifts.items():
                        if layer_idx in visual_shifts:
                            visual_shifts[layer_idx].append(shift)
                
                # Compute textual shifts (from caption differences)
                if self.text_intervention_layers:
                    t_shifts = self._compute_textual_shift(image, clean_caption, hall_caption)
                    for layer_idx, shift in t_shifts.items():
                        if layer_idx in textual_shifts:
                            textual_shifts[layer_idx].append(shift)
        
        # Apply PCA to extract principal directions for vision
        self.visual_directions = {}
        for layer_idx, shifts in visual_shifts.items():
            if len(shifts) > 0:
                shifts_matrix = torch.cat(shifts, dim=0)
                direction = self._compute_pca_direction(shifts_matrix)
                self.visual_directions[layer_idx] = direction
        
        # Apply PCA to extract principal directions for text
        self.textual_directions = {}
        for layer_idx, shifts in textual_shifts.items():
            if len(shifts) > 0:
                shifts_matrix = torch.cat(shifts, dim=0)
                direction = self._compute_pca_direction(shifts_matrix)
                self.textual_directions[layer_idx] = direction
        
        print(f"Computed {len(self.visual_directions)} visual directions and {len(self.textual_directions)} textual directions")
    
    def _compute_pca_direction(self, shifts_matrix):
        """Extract principal direction using PCA"""
        # Reshape for PCA
        if len(shifts_matrix.shape) == 3:  # [batch, seq, hidden]
            shifts_flat = shifts_matrix.reshape(-1, shifts_matrix.shape[-1])
        elif len(shifts_matrix.shape) == 2:  # [batch, hidden]
            shifts_flat = shifts_matrix
        else:
            # For 4D tensors from vision encoder [batch, channels, height, width]
            shifts_flat = shifts_matrix.reshape(shifts_matrix.shape[0], -1)
        
        shifts_flat = shifts_flat.float().cpu().numpy()
        
        # Apply PCA
        pca = PCA(n_components=1)
        pca.fit(shifts_flat)
        
        # Get principal direction
        direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        direction = direction / torch.norm(direction)
        
        return direction
    
    def _compute_visual_shift(self, image, mask_ratio, num_masks):
        """Compute shift caused by visual masking - targets vision encoder"""
        shifts = {}
        
        if not self.vision_encoder or not self.vision_layers:
            return shifts
        
        # Create clean input
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image briefly."}
            ]
        }]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process image with fixed size to avoid shape mismatches
        inputs = self.processor(
            text=prompt, 
            images=[image], 
            return_tensors="pt"
        ).to(self.model.device)
        
        # Hook to capture vision encoder activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hooks on vision encoder layers
        hooks = []
        for layer_idx in self.vision_layers:
            if hasattr(self.vision_encoder, 'encoder') and hasattr(self.vision_encoder.encoder, 'layers'):
                layer = self.vision_encoder.encoder.layers[layer_idx]
            elif hasattr(self.vision_encoder, 'layers'):
                layer = self.vision_encoder.layers[layer_idx]
            else:
                continue
            
            hook = layer.register_forward_hook(hook_fn(f"vision_layer_{layer_idx}"))
            hooks.append(hook)
        
        # Get original activations
        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True)
        
        orig_activations = {k: v.cpu() for k, v in activations.items()}
        
        # Get masked activations
        masked_activations_list = []
        for i in range(min(num_masks, 10)):  # Limit for efficiency
            masked_image = self._apply_random_mask(image, mask_ratio)
            masked_inputs = self.processor(
                text=prompt,
                images=[masked_image],
                return_tensors="pt"
            ).to(self.model.device)
            
            activations.clear()
            with torch.no_grad():
                _ = self.model(**masked_inputs, output_hidden_states=True)
            
            masked_activations_list.append({k: v.cpu() for k, v in activations.items()})
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute average shift for vision encoder
        for layer_idx in self.vision_layers:
            key = f"vision_layer_{layer_idx}"
            if key in orig_activations and any(key in ma for ma in masked_activations_list):
                orig = orig_activations[key]
                masked_tensors = [ma[key] for ma in masked_activations_list if key in ma]
                if masked_tensors:
                    masked_avg = torch.stack(masked_tensors).mean(dim=0)
                    
                    # Compute shift based on tensor shape
                    if len(orig.shape) == 4:  # Vision transformer: [batch, channels, height, width]
                        shift = (orig - masked_avg).mean(dim=[0, 2, 3])  # Average over batch, height, width
                    elif len(orig.shape) == 3:  # [batch, seq, hidden]
                        shift = (orig - masked_avg).mean(dim=[0, 1])  # Average over batch and sequence
                    else:  # [batch, hidden]
                        shift = (orig - masked_avg).mean(dim=0)
                    
                    shifts[layer_idx] = shift
        
        return shifts
    
    def _compute_textual_shift(self, image, clean_caption, hall_caption):
        """Compute shift between clean and hallucinated captions - targets text decoder"""
        shifts = {}
        
        if not self.text_decoder or not self.text_intervention_layers:
            return shifts
        
        # Create inputs for both captions
        base_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Caption: "}
            ]
        }
        
        clean_messages = [base_message, {"role": "assistant", "content": clean_caption}]
        hall_messages = [base_message, {"role": "assistant", "content": hall_caption}]
        
        clean_prompt = self.processor.apply_chat_template(clean_messages, add_generation_prompt=False)
        hall_prompt = self.processor.apply_chat_template(hall_messages, add_generation_prompt=False)
        
        clean_inputs = self.processor(text=clean_prompt, images=[image], return_tensors="pt").to(self.model.device)
        hall_inputs = self.processor(text=hall_prompt, images=[image], return_tensors="pt").to(self.model.device)
        
        # Hook to capture text decoder activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hooks on text decoder layers
        hooks = []
        for layer_idx in self.text_intervention_layers:
            layer = self.text_decoder.layers[layer_idx]
            hook = layer.register_forward_hook(hook_fn(f"text_layer_{layer_idx}"))
            hooks.append(hook)
        
        # Get activations for both captions
        with torch.no_grad():
            _ = self.model(**clean_inputs, output_hidden_states=True)
            clean_activations = {k: v.cpu() for k, v in activations.items()}
            
            activations.clear()
            _ = self.model(**hall_inputs, output_hidden_states=True)
            hall_activations = {k: v.cpu() for k, v in activations.items()}
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute shifts
        for layer_idx in self.text_intervention_layers:
            key = f"text_layer_{layer_idx}"
            if key in clean_activations and key in hall_activations:
                clean_hidden = clean_activations[key]
                hall_hidden = hall_activations[key]
                
                # Focus on the generation tokens
                if len(clean_hidden.shape) == 3:  # [batch, seq, hidden]
                    # Get the last quarter of positions (where generation happens)
                    seq_len = min(clean_hidden.shape[1], hall_hidden.shape[1])
                    gen_start = 3 * seq_len // 4
                    shift = clean_hidden[:, gen_start:, :].mean(dim=[0, 1]) - hall_hidden[:, gen_start:, :].mean(dim=[0, 1])
                else:
                    shift = clean_hidden.mean(dim=0) - hall_hidden.mean(dim=0)
                
                shifts[layer_idx] = shift
        
        return shifts
    
    def apply_interventions(self, alpha_vision: float = 0.9, alpha_text: float = 0.9):
        """
        Apply VTI interventions to both vision encoder and text decoder
        
        Args:
            alpha_vision: Intervention strength for vision encoder
            alpha_text: Intervention strength for text decoder
        """
        self.remove_interventions()
        
        if self.visual_directions is None and self.textual_directions is None:
            raise ValueError("Must compute directions before applying interventions")
        
        print(f"Applying interventions: vision alpha={alpha_vision}, text alpha={alpha_text}")
        
        # Apply vision interventions (all layers)
        if self.visual_directions and self.vision_encoder:
            self._apply_vision_interventions(alpha_vision)
        
        # Apply text interventions (selected layers)
        if self.textual_directions and self.text_decoder:
            self._apply_text_interventions(alpha_text)
    
    def _apply_vision_interventions(self, alpha):
        """Apply interventions to vision encoder"""
        def make_vision_hook(direction, alpha):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply intervention based on tensor shape
                direction_device = direction.to(hidden_states.device, dtype=hidden_states.dtype)
                
                if len(hidden_states.shape) == 4:  # [batch, channels, height, width]
                    # Reshape direction to match channels dimension
                    if direction_device.shape[0] == hidden_states.shape[1]:
                        direction_reshaped = direction_device.view(1, -1, 1, 1)
                        hidden_states = hidden_states + alpha * direction_reshaped
                elif len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                    hidden_states = hidden_states + alpha * direction_device.unsqueeze(0).unsqueeze(0)
                else:
                    hidden_states = hidden_states + alpha * direction_device
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return hook
        
        # Apply hooks to vision encoder layers
        for layer_idx, direction in self.visual_directions.items():
            if hasattr(self.vision_encoder, 'encoder') and hasattr(self.vision_encoder.encoder, 'layers'):
                layer = self.vision_encoder.encoder.layers[layer_idx]
            elif hasattr(self.vision_encoder, 'layers'):
                layer = self.vision_encoder.layers[layer_idx]
            else:
                continue
            
            hook = layer.register_forward_hook(make_vision_hook(direction, alpha))
            self.vision_hooks.append(hook)
    
    def _apply_text_interventions(self, alpha):
        """Apply interventions to text decoder"""
        def make_text_hook(direction, alpha):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply intervention
                direction_device = direction.to(hidden_states.device, dtype=hidden_states.dtype)
                
                if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                    hidden_states = hidden_states + alpha * direction_device.unsqueeze(0).unsqueeze(0)
                else:
                    hidden_states = hidden_states + alpha * direction_device
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return hook
        
        # Apply hooks to text decoder layers
        for layer_idx, direction in self.textual_directions.items():
            layer = self.text_decoder.layers[layer_idx]
            hook = layer.register_forward_hook(make_text_hook(direction, alpha))
            self.text_hooks.append(hook)
    
    def remove_interventions(self):
        """Remove all intervention hooks"""
        for hook in self.vision_hooks + self.text_hooks:
            hook.remove()
        self.vision_hooks = []
        self.text_hooks = []
    
    def _prepare_image(self, image):
        """Prepare image with correct size for the model"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get expected size from model config
        expected_size = 384  # Default for SmolVLM
        if hasattr(self.model.config, 'vision_config'):
            expected_size = self.model.config.vision_config.image_size
        
        # Resize to expected size
        image = image.resize((expected_size, expected_size), Image.LANCZOS)
        
        return image
    
    def _apply_random_mask(self, image, mask_ratio):
        """Apply random patch masking"""
        # Use processor to ensure consistent preprocessing
        if hasattr(self.processor, 'image_processor'):
            # Process image to get correct dimensions
            processed = self.processor.image_processor(image, return_tensors="pt")
            img_tensor = processed['pixel_values'][0]  # [C, H, W]
            
            # Convert back to numpy for masking
            img_array = img_tensor.permute(1, 2, 0).numpy()
            if img_array.min() < 0:  # If normalized
                img_array = ((img_array + 1) * 127.5).astype(np.uint8)
            else:
                img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = np.array(image)
        
        h, w = img_array.shape[:2]
        
        # Determine patch size based on model config
        patch_size = 14  # Default
        if hasattr(self.model.config, 'vision_config') and hasattr(self.model.config.vision_config, 'patch_size'):
            patch_size = self.model.config.vision_config.patch_size
        
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
    
    def save_directions(self, path):
        """Save computed directions"""
        torch.save({
            'visual_directions': self.visual_directions,
            'textual_directions': self.textual_directions,
            'vision_layers': self.vision_layers,
            'text_intervention_layers': self.text_intervention_layers
        }, path)
    
    def load_directions(self, path):
        """Load pre-computed directions"""
        checkpoint = torch.load(path, map_location='cpu')
        self.visual_directions = checkpoint['visual_directions']
        self.textual_directions = checkpoint['textual_directions']
        self.vision_layers = checkpoint.get('vision_layers', list(self.visual_directions.keys()) if self.visual_directions else [])
        self.text_intervention_layers = checkpoint.get('text_intervention_layers', list(self.textual_directions.keys()) if self.textual_directions else [])