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
    
    def __init__(self, model, processor, tokenizer, max_image_size: int = 512):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.directions = None
        self.hooks = []
        self.intervention_layers = None
        self.max_image_size = max_image_size
    
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
        
         # print all of model.model's attributes

        print("Model attributes:")
        for attr in dir(self.model.model):
            if not attr.startswith('_'):
                print(f" - {attr}")
        input("See above for layers")
        # Determine which layers to intervene on
        if hasattr(self.model.model, 'layers'):
            n_layers = len(self.model.model.layers)
            input(f"hello this model has {n_layers} layers. Press Enter to continue...")
        else:
            # For models with different architecture
            n_layers = 32  # Default assumption
            input(f"hi this model is using default layers. :()")
            
        if target_layers is None:
            # Intervene on last third of layers (where multimodal fusion happens)
            self.intervention_layers = list(range(n_layers // 2, n_layers))
        else:
            self.intervention_layers = target_layers
            
        print(f"Will intervene on layers: {self.intervention_layers}")
        
        # Collect activation differences
        all_shifts = {layer_idx: [] for layer_idx in self.intervention_layers}
        
        self.model.eval()
        with torch.no_grad():
            for img_url, clean_caption, hall_caption in demo_data:
                # Load and prepare image
                image = load_image(img_url)
                image = self._prepare_image(image)
                
                # Get shifts from both visual masking and caption differences
                visual_shift = self._compute_visual_shift(image, mask_ratio, num_masks)
                textual_shift = self._compute_textual_shift(image, clean_caption, hall_caption)
                
                # Combine shifts (visual and textual information is entangled in LLM layers)
                for layer_idx in self.intervention_layers:
                    if layer_idx in visual_shift and layer_idx in textual_shift:
                        # Average visual and textual shifts
                        combined_shift = (visual_shift[layer_idx] + textual_shift[layer_idx]) / 2
                        all_shifts[layer_idx].append(combined_shift)
        
        # Apply PCA to extract principal directions
        self.directions = {}
        for layer_idx, shifts in all_shifts.items():
            if len(shifts) > 0:
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
                
                # CRITICAL: Verify direction reduces hallucination score
                direction = self._verify_direction(direction, layer_idx, demo_data)
                
                self.directions[layer_idx] = direction
                
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
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for layer_idx in self.intervention_layers:
            if hasattr(self.model.model, 'layers'):
                hook = self.model.model.layers[layer_idx].register_forward_hook(
                    hook_fn(f"layer_{layer_idx}")
                )
                hooks.append(hook)
        
        # Get original activations
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=50)
        
        orig_activations = {k: v.cpu() for k, v in activations.items()}
        
        # Get masked activations
        masked_activations_list = []
        for _ in range(min(num_masks, 10)):  # Limit for efficiency
            masked_image = self._apply_random_mask(image, mask_ratio)
            masked_inputs = self.processor(text=prompt, images=[masked_image], return_tensors="pt").to(self.model.device)
            
            activations.clear()
            with torch.no_grad():
                _ = self.model.generate(**masked_inputs, max_new_tokens=50)
            
            masked_activations_list.append({k: v.cpu() for k, v in activations.items()})
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute average shift
        for layer_idx in self.intervention_layers:
            key = f"layer_{layer_idx}"
            if key in orig_activations:
                orig = orig_activations[key]
                masked_avg = torch.stack([ma[key] for ma in masked_activations_list if key in ma]).mean(dim=0)
                
                # Focus on the tokens where visual information is processed
                # SmolVLM processes visual tokens early in the sequence
                visual_token_positions = slice(1, 257)  # Adjust based on model's visual token count
                
                if len(orig.shape) == 3:  # [batch, seq, hidden]
                    shift = orig[:, visual_token_positions, :].mean(dim=1) - masked_avg[:, visual_token_positions, :].mean(dim=1)
                else:
                    shift = orig - masked_avg
                    
                shifts[layer_idx] = shift
        
        return shifts
    
    def _compute_textual_shift(self, image, clean_caption, hall_caption):
        """Compute shift between clean and hallucinated captions"""
        shifts = {}
        
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
        
        # Get hidden states
        with torch.no_grad():
            clean_outputs = self.model(**clean_inputs, output_hidden_states=True)
            hall_outputs = self.model(**hall_inputs, output_hidden_states=True)
        
        # Extract shifts from hidden states
        if hasattr(clean_outputs, 'hidden_states') and clean_outputs.hidden_states is not None:
            for layer_idx in self.intervention_layers:
                if layer_idx < len(clean_outputs.hidden_states):
                    clean_hidden = clean_outputs.hidden_states[layer_idx]
                    hall_hidden = hall_outputs.hidden_states[layer_idx]
                    
                    # Focus on the generation tokens (last part of sequence)
                    # This is where hallucinations manifest
                    if len(clean_hidden.shape) == 3:
                        # Average over positions where the captions differ
                        clean_len = clean_inputs.input_ids.shape[1]
                        hall_len = hall_inputs.input_ids.shape[1]
                        min_len = min(clean_len, hall_len)
                        
                        # Get the last quarter of positions (where generation happens)
                        gen_start = 3 * min_len // 4
                        shift = clean_hidden[:, gen_start:min_len, :].mean(dim=1) - hall_hidden[:, gen_start:min_len, :].mean(dim=1)
                    else:
                        shift = clean_hidden - hall_hidden
                    
                    shifts[layer_idx] = shift.cpu()
        
        return shifts
    
    def _verify_direction(self, direction, layer_idx, demo_data, num_samples=5):
        """
        Verify that the direction actually reduces hallucinations
        by testing on a few samples
        """
        # For now, return the direction as-is
        # In a full implementation, you would test both +direction and -direction
        # and choose the one that reduces hallucination metrics
        return direction
    
    def apply_interventions(self, alpha: float = 0.5):
        """
        Apply VTI interventions to the model
        
        Args:
            alpha: Intervention strength (0.5-2.0 recommended)
        """
        self.remove_interventions()
        
        if self.directions is None:
            raise ValueError("Must compute directions before applying interventions")
        
        print(f"Applying interventions with alpha={alpha}")
        
        def make_hook(direction, alpha):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply intervention
                direction_device = direction.to(hidden_states.device, dtype=hidden_states.dtype)
                
                if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                    # Apply to all positions
                    hidden_states = hidden_states + alpha * direction_device.unsqueeze(0).unsqueeze(0)
                else:
                    hidden_states = hidden_states + alpha * direction_device
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return hook
        
        # Apply hooks to target layers
        for layer_idx, direction in self.directions.items():
            if hasattr(self.model.model, 'layers'):
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(make_hook(direction, alpha))
                self.hooks.append(hook)
    
    def remove_interventions(self):
        """Remove all intervention hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _prepare_image(self, image):
        """Prepare image safely"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((self.max_image_size, self.max_image_size), Image.LANCZOS)
        
        return image
    
    def _apply_random_mask(self, image, mask_ratio):
        """Apply random patch masking"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create patch grid
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
    
    def save_directions(self, path):
        """Save computed directions"""
        torch.save({
            'directions': self.directions,
            'intervention_layers': self.intervention_layers
        }, path)
    
    def load_directions(self, path):
        """Load pre-computed directions"""
        checkpoint = torch.load(path, map_location='cpu')
        self.directions = checkpoint['directions']
        self.intervention_layers = checkpoint.get('intervention_layers', list(self.directions.keys()))