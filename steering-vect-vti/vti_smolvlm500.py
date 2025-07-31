"""
Example script showing how to use VTI with SmolVLM-500M-Instruct (Idefics3 architecture)
"""

import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from vti_implementation import VTI
import requests
from io import BytesIO
import json

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# VTI hyperparameters - adjusted for 500M model
ALPHA = 0.8  # Increased strength for 500M model
MASK_RATIO = 0.95   # Slightly reduced mask ratio
NUM_MASKS = 75      # Increased number of masks for better averaging

def load_vti_demo_data():
    """Load demonstration data for VTI direction computation"""
    data = [json.loads(line) for line in open('hallucination_vti_demos.jsonl')]
    return [("http://images.cocodataset.org/train2014/" + d['image'], d['value'], d['h_value']) for d in data]

def explore_idefics3_architecture(model):
    """Explore the Idefics3 model architecture to understand its structure"""
    print("=== Idefics3 Model Architecture Analysis ===")
    
    # Main model components
    print("\nMain model components:")
    for attr in ['model', 'lm_head']:
        if hasattr(model, attr):
            component = getattr(model, attr)
            print(f"  {attr}: {type(component)}")
    
    # Explore the core model
    if hasattr(model, 'model'):
        core_model = model.model
        print(f"\nCore model type: {type(core_model)}")
        
        # Check for vision and text components
        vision_attrs = ['vision_model', 'vision_encoder', 'vision_tower']
        text_attrs = ['text_model', 'language_model', 'decoder']
        
        print("\nVision components:")
        for attr in vision_attrs:
            if hasattr(core_model, attr):
                component = getattr(core_model, attr)
                print(f"  ✓ {attr}: {type(component)}")
                
                # Check for layers in vision component
                if hasattr(component, 'layers'):
                    print(f"    - Has {len(component.layers)} layers")
                elif hasattr(component, 'encoder') and hasattr(component.encoder, 'layers'):
                    print(f"    - encoder.layers: {len(component.encoder.layers)}")
        
        print("\nText/Language components:")
        for attr in text_attrs:
            if hasattr(core_model, attr):
                component = getattr(core_model, attr)
                print(f"  ✓ {attr}: {type(component)}")
                
                # Check for layers
                if hasattr(component, 'layers'):
                    print(f"    - Has {len(component.layers)} layers")
    
    # Check model config for architecture details
    if hasattr(model, 'config'):
        config = model.config
        print(f"\nModel config:")
        print(f"  Architecture: {getattr(config, 'architectures', 'unknown')}")
        print(f"  Model type: {getattr(config, 'model_type', 'unknown')}")
        
        # Vision config details
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            print(f"  Vision layers: {getattr(vision_config, 'num_hidden_layers', 'unknown')}")
            print(f"  Vision hidden size: {getattr(vision_config, 'hidden_size', 'unknown')}")
            print(f"  Patch size: {getattr(vision_config, 'patch_size', 'unknown')}")

class Idefics3VTI(VTI):
    """Extended VTI class specifically for Idefics3 architecture"""
    
    def __init__(self, model, processor, tokenizer):
        super().__init__(model, processor, tokenizer)
        self._setup_idefics3_paths()
    
    def _setup_idefics3_paths(self):
        """Setup the correct paths for Idefics3 architecture"""
        # Try to find the vision encoder in Idefics3
        vision_paths = [
            'model.vision_model',
            'model.vision_encoder', 
            'model.vision_tower'
        ]
        
        self.vision_encoder = None
        for path in vision_paths:
            try:
                obj = self.model
                for part in path.split('.'):
                    obj = getattr(obj, part)
                self.vision_encoder = obj
                self.vision_path = path
                print(f"Found vision encoder at: {path}")
                break
            except AttributeError:
                continue
        
        if self.vision_encoder is None:
            raise ValueError("Could not find vision encoder in Idefics3 model")
        
        # Find vision layers
        if hasattr(self.vision_encoder, 'layers'):
            self.vision_layers = self.vision_encoder.layers
            print(f"Found {len(self.vision_layers)} vision layers")
        elif hasattr(self.vision_encoder, 'encoder') and hasattr(self.vision_encoder.encoder, 'layers'):
            self.vision_layers = self.vision_encoder.encoder.layers
            print(f"Found {len(self.vision_layers)} vision layers in encoder")
        else:
            raise ValueError("Could not find vision layers in Idefics3 model")
    
    def get_vision_representations(self, images, mask_patches=False, mask_ratio=0.99):
        """Get vision representations with optional patch masking for Idefics3"""
        with torch.no_grad():
            # Process images through the vision encoder
            if hasattr(self.vision_encoder, 'patch_embed'):
                # Direct patch embedding approach
                pixel_values = images
                if len(pixel_values.shape) == 5:  # Batch dimension handling
                    pixel_values = pixel_values.squeeze(0)
                
                # Get patch embeddings
                patch_embeds = self.vision_encoder.patch_embed(pixel_values)
                
                if mask_patches:
                    # Apply patch masking
                    B, N, D = patch_embeds.shape
                    num_masked = int(N * mask_ratio)
                    
                    # Create random mask
                    mask = torch.rand(B, N, device=patch_embeds.device)
                    mask_indices = mask.topk(num_masked, dim=1)[1]
                    
                    # Apply mask
                    masked_embeds = patch_embeds.clone()
                    for b in range(B):
                        masked_embeds[b, mask_indices[b]] = 0
                    
                    patch_embeds = masked_embeds
                
                # Pass through transformer layers
                hidden_states = patch_embeds
                for layer in self.vision_layers:
                    hidden_states = layer(hidden_states)[0] if isinstance(layer(hidden_states), tuple) else layer(hidden_states)
                
                return hidden_states
            
            else:
                # Alternative approach for different architectures
                # Use the full vision encoder forward pass
                if mask_patches:
                    # This is more complex for full forward pass
                    # We'll need to hook into intermediate representations
                    return self._get_masked_vision_features(images, mask_ratio)
                else:
                    return self.vision_encoder(images).last_hidden_state if hasattr(self.vision_encoder(images), 'last_hidden_state') else self.vision_encoder(images)
    
    def _get_masked_vision_features(self, images, mask_ratio):
        """Get vision features with masking applied at patch level"""
        # This is a simplified approach - you might need to adapt based on exact architecture
        features = self.vision_encoder(images)
        if hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state
        
        # Apply masking to features
        B, N, D = features.shape
        num_masked = int(N * mask_ratio)
        
        mask = torch.rand(B, N, device=features.device)
        mask_indices = mask.topk(num_masked, dim=1)[1]
        
        masked_features = features.clone()
        for b in range(B):
            masked_features[b, mask_indices[b]] = 0
        
        return masked_features

def main():
    print("Loading SmolVLM-500M model with Idefics3 architecture...")
    
    # Load model components with proper configuration
    max_image_size = 512
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        max_image_size={"longest_edge": max_image_size}
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else torch.float16,
        device_map={"": DEVICE}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Explore architecture before creating VTI
    explore_idefics3_architecture(model)
    
    # Create specialized VTI handler for Idefics3
    print("\nInitializing Idefics3-specific VTI handler...")
    try:
        vti = Idefics3VTI(model, processor, tokenizer)
    except Exception as e:
        print(f"Error creating Idefics3VTI: {e}")
        print("Falling back to standard VTI...")
        vti = VTI(model, processor, tokenizer)
    
    # Load demonstration data
    demo_data = load_vti_demo_data()
    print(f"\nLoaded {len(demo_data)} demonstration examples for VTI.")
    
    # Compute VTI directions with updated parameters
    print("Computing VTI directions for Idefics3...")
    try:
        vti.compute_directions(
            demo_data=demo_data,
            mask_ratio=MASK_RATIO,
            num_masks=NUM_MASKS
        )
        
        # Save directions
        directions_file = "idefics3_vti_directions_smolvlm_500m.pt"
        vti.save_directions(directions_file)
        print(f"Saved VTI directions to {directions_file}")
        
    except Exception as e:
        print(f"Error computing directions: {e}")
        print("This might be due to architecture differences. Consider debugging the vision encoder structure.")
        return
    
    # Test with different alpha values
    test_alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Prepare test image
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    response = requests.get(test_image_url)
    test_image = Image.open(BytesIO(response.content))
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to(DEVICE)
    
    # Generate baseline without VTI
    print("\n" + "="*50)
    print("BASELINE (No VTI):")
    print("="*50)
    
    vti.remove_interventions()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    baseline_output = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    print(baseline_output)
    
    # Test different alpha values
    for alpha in test_alphas:
        print(f"\n" + "="*50)
        print(f"VTI with α = {alpha}:")
        print("="*50)
        
        try:
            vti.apply_interventions(alpha=alpha)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            vti_output = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            print(vti_output)
            
            # Check if output differs from baseline
            if vti_output.strip() != baseline_output.strip():
                print(f"✓ VTI effect detected with α = {alpha}")
            else:
                print(f"⚠ No difference from baseline with α = {alpha}")
            
        except Exception as e:
            print(f"Error with α = {alpha}: {e}")
        
        finally:
            vti.remove_interventions()
    
    print(f"\n" + "="*50)
    print("VTI testing complete!")
    print("="*50)

if __name__ == "__main__":
    main()