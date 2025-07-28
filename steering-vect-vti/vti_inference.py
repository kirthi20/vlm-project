"""
Example script showing how to use VTI with SmolVLM-256M-Instruct
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
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# VTI hyperparameters (from the paper)
ALPHA_VISION = 0.9  # Strength of visual intervention
ALPHA_TEXT = 0.9    # Strength of textual intervention
MASK_RATIO = 0.99   # Ratio of patches to mask
NUM_MASKS = 50      # Number of mask perturbations to average
NUM_DEMOS = 50      # Number of demonstration examples

def load_vti_demo_data():
    data = [json.loads(line) for line in open('hallucination_vti_demos.jsonl')]
    return [("http://images.cocodataset.org/train2014/" + d['image'], d['value'], d['h_value']) for d in data]

# Debug snippet to find the vision encoder path
def find_vision_encoder(model): # DEBUG
    print("Model type:", type(model))
    print("\nModel attributes:")
    for attr in dir(model):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    # Common paths to check
    possible_paths = [
        'model.vision_model',
        'model.vision_tower', 
        'vision_model',
        'vision_tower',
        'model.encoder',
        'encoder'
    ]
    
    for path in possible_paths:
        try:
            obj = model
            for part in path.split('.'):
                obj = getattr(obj, part)
            print(f"\n✓ Found at: {path}")
            print(f"  Type: {type(obj)}")
            if hasattr(obj, 'layers'):
                print(f"  Has layers: {len(obj.layers)}")
        except AttributeError:
            continue
    
    # Check model config
    if hasattr(model, 'config'):
        print("\nModel config type:", type(model.config))


def find_language_model(model):
    print("Looking for language model...")
    
    # Check direct attributes
    if hasattr(model, 'language_model'):
        print("Found at: model.language_model")
    elif hasattr(model, 'text_model'):
        print("Found at: model.text_model")
    elif hasattr(model.model, 'text_model'):
        print("Found at: model.model.text_model")
    elif hasattr(model.model, 'language_model'):
        print("Found at: model.model.language_model")
    
    # For Idefics3, it's likely:
    if hasattr(model.model, 'decoder'):
        print("Found decoder at: model.model.decoder")
        if hasattr(model.model.decoder, 'layers'):
            print(f"  Has {len(model.model.decoder.layers)} layers")
    
    # Check model.model attributes
    print("\nmodel.model attributes:")
    for attr in dir(model.model):
        if not attr.startswith('_') and 'model' in attr or 'decoder' in attr or 'text' in attr:
            print(f"  {attr}")

def main():
    print("Loading SmolVLM model...")
    # Load model components
    # Load processor with explicit max_image_size using the correct format
    max_image_size = 512  # Use the model's native patch size

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        max_image_size={"longest_edge": max_image_size}  # Use dictionary format
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else torch.float16 if DEVICE == "mps" else torch.float32,
        device_map={"":DEVICE}
    )

    find_language_model(model)
    input()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Create VTI handler
    vti = VTI(model, processor, tokenizer)
    
    demo_data = load_vti_demo_data()

    print(f"Loaded {len(demo_data)} demonstration examples for VTI.")
    print("Demonstration data sample:", demo_data[0])
    
    print("Computing VTI directions...")
    vti.compute_directions(
        demo_data=demo_data,
        mask_ratio=MASK_RATIO,
        num_masks=NUM_MASKS
    )
    
    # Save directions for future use
    vti.save_directions("vti_directions_smolvlm.pt")
    
    # Option 2: Load pre-computed directions
    # vti.load_directions("vti_directions_smolvlm.pt")
    
    # Apply interventions
    print("Applying VTI interventions...")
    vti.apply_interventions(alpha_vision=ALPHA_VISION, alpha_text=ALPHA_TEXT)
    
    # Test the model with VTI
    print("\nTesting model with VTI...")
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    
    # Download test image
    response = requests.get(test_image_url)
    test_image = Image.open(BytesIO(response.content))
    
    # Generate caption without VTI
    vti.remove_interventions()
    print("\nWithout VTI:")
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
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    
    output_no_vti = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    print(output_no_vti)
    
    # Generate caption with VTI
    vti.apply_interventions(alpha_vision=ALPHA_VISION, alpha_text=ALPHA_TEXT)
    print("\nWith VTI:")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    
    output_with_vti = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    print(output_with_vti)
    
    # Clean up
    vti.remove_interventions()
    
    print("\nVTI setup complete!")


if __name__ == "__main__":
    # Run the full example
    main()
    
    # Or run the minimal example
    # minimal_example()