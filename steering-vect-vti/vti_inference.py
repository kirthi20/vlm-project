"""
Example script showing how to use VTI with SmolVLM-256M-Instruct
Corrected to apply separate interventions to vision encoder and text decoder
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
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

# VTI hyperparameters (from the paper)
ALPHA_VISION = 0.3  # Strength of visual intervention
ALPHA_TEXT = 0.4    # Strength of textual intervention
MASK_RATIO = 0.99   # Ratio of patches to mask
NUM_MASKS = 50      # Number of mask perturbations

def load_vti_demo_data():
    """Load demonstration data for VTI"""
    data = [json.loads(line) for line in open('hallucination_vti_demos.jsonl')]
    return [("http://images.cocodataset.org/train2014/" + d['image'], d['value'], d['h_value']) for d in data]

def explore_model_structure(model):
    """Helper to understand model structure"""
    print("\n" + "="*50)
    print("EXPLORING MODEL STRUCTURE:")
    print("="*50)
    
    # Print main model structure
    print(f"\nModel type: {type(model)}")
    print(f"\nModel config vision size: {model.config.vision_config.image_size if hasattr(model.config, 'vision_config') else 'Unknown'}")
    
    # Look for vision and text components
    for name, module in model.named_children():
        print(f"\nTop-level component: {name} -> {type(module)}")
        if hasattr(module, 'vision_tower'):
            print(f"  - Has vision_tower")
        if hasattr(module, 'text_model'):
            print(f"  - Has text_model") 
        if hasattr(module, 'language_model'):
            print(f"  - Has language_model")
        if hasattr(module, 'layers'):
            print(f"  - Has {len(module.layers)} layers")

def main():
    print("Loading SmolVLM model...")
    
    # Load processor without specifying size to use model defaults
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else torch.float32,
        device_map={"": DEVICE}
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Explore model structure
    explore_model_structure(model)
    
    # Create VTI handler
    print("\nInitializing VTI...")
    vti = VTI(model, processor, tokenizer)
    
    # Load demonstration data
    demo_data = load_vti_demo_data()
    print(f"\nLoaded {len(demo_data)} demonstration examples for VTI.")
    
    # Use a subset for efficiency during development
    demo_subset = demo_data[:5]  # VTI paper uses 50 examples
    print(f"Using {len(demo_subset)} samples for direction computation")
    
    # Option 1: Compute directions
    print("\nComputing VTI directions...")
    vti.compute_directions(
        demo_data=demo_subset,
        mask_ratio=MASK_RATIO,
        num_masks=NUM_MASKS
    )
    
    # Save directions for future use
    vti.save_directions("smolvlm_256m_vti_directions.pt")
    print("Saved VTI directions to smolvlm_256m_vti_directions.pt")
    
    # Option 2: Load pre-computed directions (uncomment to use)
    # vti.load_directions("smolvlm_256m_vti_directions.pt")
    
    # Test the model with and without VTI
    print("\nTesting model with VTI...")
    test_image_url = "http://images.cocodataset.org/train2014/COCO_train2014_000000103108.jpg"
    
    # Download test image
    response = requests.get(test_image_url)
    test_image = Image.open(BytesIO(response.content))
    
    # Create test messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail. What objects can you see?"}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to(DEVICE)
    
    # Check processed image shape
    if 'pixel_values' in inputs:
        print(f"\nProcessed image shape: {inputs['pixel_values'].shape}")
    
    # Generate caption without VTI
    print("\n" + "="*50)
    print("WITHOUT VTI:")
    print("="*50)
    
    vti.remove_interventions()  # Ensure no interventions are active
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=False,  # Deterministic generation
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    input_length = inputs.input_ids.shape[1]
    generated_tokens = generated_ids[:, input_length:]
    
    output_no_vti = processor.batch_decode(
        generated_tokens, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    print(output_no_vti)
    
    # Generate caption with VTI
    print("\n" + "="*50)
    print(f"WITH VTI (vision alpha={ALPHA_VISION}, text alpha={ALPHA_TEXT}):")
    print("="*50)
    
    vti.apply_interventions(alpha_vision=ALPHA_VISION, alpha_text=ALPHA_TEXT)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_tokens = generated_ids[:, input_length:]
    
    output_with_vti = processor.batch_decode(
        generated_tokens, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    print(output_with_vti)
    
    # Test with different alpha combinations
    print("\n" + "="*50)


if __name__ == "__main__":
    main()