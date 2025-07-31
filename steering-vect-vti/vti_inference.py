"""
Example script showing how to use VTI with SmolVLM-500M-Instruct
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
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct" #"HuggingFaceTB/SmolVLM-500M-Instruct"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

# VTI hyperparameters (from the paper)
ALPHA = 0.6  # Increased strength of visual intervention
MASK_RATIO = 0.99   # Ratio of patches to mask
NUM_MASKS = 20      # Reduced for efficiency

def load_vti_demo_data():
    """Load demonstration data for VTI"""
    data = [json.loads(line) for line in open('hallucination_vti_demos.jsonl')]
    return [("http://images.cocodataset.org/train2014/" + d['image'], d['value'], d['h_value']) for d in data]

def main():
    print("Loading SmolVLM model...")
    
    # Load processor with explicit max_image_size
    max_image_size = 384  # Smaller for efficiency
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        max_image_size={"longest_edge": max_image_size}
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE.startswith('cuda') else torch.float32,
        device_map={"": DEVICE}
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Create VTI handler
    print("Initializing VTI...")
    vti = VTI(model, processor, tokenizer)
    
    # Load demonstration data
    demo_data = load_vti_demo_data()
    print(f"Loaded {len(demo_data)} demonstration examples for VTI.")
    
    # Use a subset for efficiency during development
    demo_subset = demo_data[:10]  # Use first 10 samples
    #print(f"Using {len(demo_subset)} samples for direction computation")
    
    print("Computing VTI directions...")
    vti.compute_directions(
        demo_data=demo_subset,
        mask_ratio=MASK_RATIO,
        num_masks=NUM_MASKS
    )
    
    # Save directions for future use
    vti.save_directions("smolvlm_500m_vti_directions.pt")
    
    # Option 2: Load pre-computed directions (uncomment to use)
    # vti.load_directions("smolvlm_500m_vti_directions.pt")
    
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
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to(DEVICE)
    
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
    print(f"WITH VTI (alpha={ALPHA}):")
    print("="*50)
    
    vti.apply_interventions(alpha=ALPHA)
    
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
    
    # Test with different alpha values
    print("\n" + "="*50)
    print("TESTING DIFFERENT ALPHA VALUES:")
    print("="*50)
    
    for alpha in [0.5, 1.5, 2.0]:
        print(f"\nAlpha = {alpha}:")
        print("-" * 30)
        
        vti.remove_interventions()
        vti.apply_interventions(alpha=alpha)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = generated_ids[:, input_length:]
        output = processor.batch_decode(
            generated_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        print(output)
    
    # Clean up
    vti.remove_interventions()
    
    print("\n" + "="*50)
    print("VTI setup and testing complete!")
    print("="*50)


if __name__ == "__main__":
    main()