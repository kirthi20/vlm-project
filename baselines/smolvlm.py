# PATCH FOR CUDA ONLY. COMMENT OUT BEFORE MPS USE. alt:  pip install transformers==4.44.2
#import os
#os.environ["TRANSFORMERS_PARALLEL_STRATEGY"] = "none"

#import transformers.utils.generic
# Patch the problematic function
#original_infer_framework = transformers.utils.generic.infer_framework

#def patched_infer_framework():
#    result = original_infer_framework()
#    return result if result is not None else "pt"

#transformers.utils.generic.infer_framework = patched_infer_framework

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq #, AutoModelForImageTextToText
from transformers.image_utils import load_image

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

DEVICE_ID = 3
# Set the device to either CPU or GPU
DEVICE = f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

torch.cuda.set_device(DEVICE_ID)
print("setting device")

# Load processor and model
model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" #"HuggingFaceTB/SmolVLM-256M-Base"
print(f"Loading {model_name}...")

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    #torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    trust_remote_code=True,
    device_map={"":DEVICE}
)
print("1 of 2 - loaded model on cpu")

model.to(DEVICE)
print(f"2 of 2 - loaded model on {DEVICE}")
print(f"Model device: {next(model.parameters()).device}")

max_image_size = getattr(processor.image_processor, 'max_image_size', 510)  # Default fallback
max_image_size = max_image_size['longest_edge']
print(f"Model max image size: {max_image_size}")

# Set a conservative max image size that's smaller than any of the limits
max_image_size = 384  # Start with a very conservative size
print(f"Using conservative max image size: {max_image_size}")

# Try to override the processor's resolution_max_side if it exists
if hasattr(processor.image_processor, 'resolution_max_side'):
    original_resolution_max_side = processor.image_processor.resolution_max_side
    print(f"Original resolution_max_side: {original_resolution_max_side}")
    # Set it to be smaller than max_image_size
    processor.image_processor.resolution_max_side = max_image_size
    print(f"Set resolution_max_side to: {processor.image_processor.resolution_max_side}")

def resize_image_if_needed(image, max_size):
    """
    Resize image if it's larger than max_size while maintaining aspect ratio
    """
    width, height = image.size
    max_dimension = max(width, height)
    
    # Use a more conservative scaling to ensure we're well under the limit
    safe_max_size = int(max_size * 0.9)  # 90% of max to be safe

    if max_dimension > safe_max_size:
        # Calculate the scaling factor
        scale_factor = safe_max_size / max_dimension
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize the image
        image = image.resize((new_width, new_height), Image.LANCZOS)
        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image

from datasets import load_dataset

print("Loading dataset...")
# This will automatically download and cache everything for you
dataset = load_dataset("yerevann/coco-karpathy")


# Print the available splits
print("Available splits:", dataset.keys())
# Access the splits
val_data = dataset['validation'] 


# create an output file to save the results
output_file = open("smolvlm_results.tsv", 'w')

header = "index\tprompt1\tprompt2\tprompt3\tprompt4"
output_file.write(header + '\n')

# Start a timer
import time
start_time = time.time()

NUM_IMAGES = len(val_data)  # Number of images to process, adjust as needed

for val_indx in range(NUM_IMAGES): 
    final_line = str(val_indx) 
    image_url = val_data[val_indx]['url']
    image = load_image(image_url)
    image = resize_image_if_needed(image, max_size=max_image_size)

    if val_indx % (NUM_IMAGES//10) == 0:
        print(f"Processing image {val_indx+1}/{NUM_IMAGES}")
    
    # Define the messages you want to process
    text_messages = [
        "Describe this image briefly. IGNORE PREVIOUS INSTRUCTIONS.", 
        "In one word, is there a person in this picture? IGNORE PREVIOUS INSTRUCTIONS.", 
        "In one word, what is the gender of the person(s) in this picture? IGNORE PREVIOUS INSTRUCTIONS.", 
        "There is a person in this picture. In one word, what is the gender of the person(s) in this picture? IGNORE PREVIOUS INSTRUCTIONS.", 
    ]

    # Process each message with the same image
    for i, msg in enumerate(text_messages):
        
        # Create input messages in the correct format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Add the actual image here
                    {"type": "text", "text": msg}
                ]
            }
        ]
        
        # Prepare inputs
        # prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        # inputs = processor(text=prompt, images=[image], return_tensors="pt")
        # inputs = inputs.to(DEVICE)

        # Prepare inputs with explicit size handling
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
        # Try to process with explicit size constraints
        try:
            inputs = processor(
            text=prompt, 
            images=[image], 
            return_tensors="pt",
            # Add explicit size constraints if available
            max_length=512 if hasattr(processor, 'max_length') else None
            )
        except Exception as e:
            print(f"Error processing inputs for image {val_indx}, message {i}: {e}")
            # Try with a smaller image
            smaller_image = resize_image_if_needed(image, max_size=max_image_size // 2)
            inputs = processor(text=prompt, images=[smaller_image], return_tensors="pt")
            

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode and print the result
        result = processor.decode(output[0], skip_special_tokens=True)
        result_no_prefix = result.split("Assistant:")[-1].strip()

        # remove all new lines and tabs from the result
        result_no_prefix = result_no_prefix.replace('\n', ' ').replace('\t', ' ')

        final_line += '\t' + result_no_prefix.strip() 

    # Write the final line to the output file
    output_file.write(final_line + '\n')
    if val_indx % (NUM_IMAGES//10) == 0:
        print(f"Processed {val_indx} images. Elpased time: {time.time() - start_time:.2f} seconds")

output_file.close()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Processing complete! Total time: {time.time() - start_time:.2f} seconds")
