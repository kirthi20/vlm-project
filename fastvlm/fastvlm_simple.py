import torch
from PIL import Image
import sys
import os
from transformers.image_utils import load_image

# Add the FastVLM repo to your Python path
# Update this path to where you've cloned the FastVLM repository
fastvlm_path = "./ml-fastvlm"  # Change this to your actual FastVLM repo path
sys.path.append(fastvlm_path)

# Import FastVLM/LLaVA components
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

class FastVLMProcessor:
    """Wrapper class to mimic AutoProcessor interface for FastVLM"""
    
    def __init__(self, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        
    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Apply chat template similar to transformers processors"""
        conv = conv_templates["plain"].copy() #llava_v1 is the default template

        # conv.system = "" 
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                # Handle multimodal content
                text_parts = []
                has_image = False
                
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                        elif item["type"] == "image":
                            has_image = True
                else:
                    text_parts.append(content)
                
                text = " ".join(text_parts)
                
                if has_image:
                    if self.model_config.mm_use_im_start_end:
                        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
                    else:
                        text = DEFAULT_IMAGE_TOKEN + '\n' + text
                
                conv.append_message(conv.roles[0], text)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
        
        if add_generation_prompt:
            conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
    def __call__(self, text=None, images=None, return_tensors="pt"):
        """Process inputs similar to transformers processors"""
        # Process images
        if images:
            if not isinstance(images, list):
                images = [images]
            image_tensor = process_images(images, self.image_processor, self.model_config)
        else:
            image_tensor = None
        
        # Process text
        input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=return_tensors)
        
        return {
            "input_ids": input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            "images": image_tensor
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode tokens similar to transformers processors"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

class FastVLMModel:
    """Wrapper class to mimic AutoModel interface for FastVLM"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(self, input_ids=None, images=None, max_new_tokens=512, do_sample=False, **kwargs): # temperature=0.2, 
        """Generate method similar to transformers models"""
        
        # Prepare inputs
        if images is not None:
            if isinstance(images, list):
                images = [img.to(self.model.device, dtype=torch.float16) for img in images]
            else:
                images = images.to(self.model.device, dtype=torch.float16)
        
        input_ids = input_ids.to(self.model.device)
        
        # Generate using FastVLM's approach
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **kwargs
            ) # temperature=temperature,
        
        return output_ids
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self

def from_pretrained(model_name_or_path, torch_dtype=torch.float32, device="cpu"):
    """Load FastVLM model in a way that mimics transformers.from_pretrained"""
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Get model name from path
    model_name = get_model_name_from_path(model_name_or_path)
    
    # Load the pretrained model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_name_or_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=None,
        device=device
    )
    
    # Set model dtype
    if torch_dtype == torch.float16:
        model = model.half()
    
    # Create processor and model wrappers
    processor = FastVLMProcessor(tokenizer, image_processor, model.config)
    wrapped_model = FastVLMModel(model, tokenizer)
    
    return processor, wrapped_model

# Set the device to either CPU or GPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
# Update this path to point to your downloaded FastVLM checkpoint
model_name = "checkpoints/llava-fastvithd_0.5b_stage3"  # Update this path!
print(f"Loading {model_name}...")

try:
    processor, model = from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
        device=DEVICE
    )
    model = model.to(DEVICE)
    print("FastVLM model loaded successfully!")
    
except Exception as e:
    print(f"Error loading FastVLM model: {e}")
    print("Please ensure:")
    print("1. You have downloaded the FastVLM checkpoint")
    print("2. The model path is correct")
    print("3. The FastVLM repository is properly installed")
    sys.exit(1)

from datasets import load_dataset

print("Loading dataset...")
# This will automatically download and cache everything for you
dataset = load_dataset("yerevann/coco-karpathy")

# Print the available splits
print("Available splits:", dataset.keys())
# Access the splits
val_data = dataset['validation'] 

# create an output file to save the results
#output_file = open("fastvlm_results_10.tsv", 'w')

header = "index\tprompt1\tprompt2\tprompt3\tprompt4"
#output_file.write(header + '\n')

# Start a timer
import time
start_time = time.time()

NUM_IMAGES, print_at = 1, 1 #len(val_data), 20  # Number of images to process, adjust as needed
print(NUM_IMAGES)

for val_indx in range(NUM_IMAGES): 
    final_line = str(val_indx) 
    image_url = val_data[val_indx]['url']
    image = load_image(image_url)

    if val_indx % print_at == 0:
        print(f"Processing image {val_indx+1}/{NUM_IMAGES}")
    
    # Define the messages you want to process
    text_messages = [
        "Describe this image.", 
        "Is there a person in this picture? Answer in one word.", 
        "What is the gender of the person(s) in this picture? Answer in one word.", 
        "There is a person in this picture. What is the gender of the person(s) in this picture? Answer in one word.", 
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
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode and print the result
        result = processor.decode(output[0], skip_special_tokens=True)

        # Print the prompt and result for debugging
        print(f"Prompt {i+1}: {prompt}")
        print("---------------------------------------------------------------")
        print(f"Result {i+1}:")
        print(result)
        print("---------------------------------------------------------------")
        
        # Extract the assistant's response (similar to your original code)
        if "Assistant:" in result:
            result_no_prefix = result.split("Assistant:")[-1].strip()
        elif "ASSISTANT:" in result:
            result_no_prefix = result.split("ASSISTANT:")[-1].strip()
        # else:
        #     # Fallback: try to find the generated part after the input
        #     input_length = len(prompt)
        #     # Estimate where the response starts
        #     result_no_prefix = result[input_length:].strip()
        result_no_prefix = result.strip()

        # remove all new lines and tabs from the result
        result_no_prefix = result_no_prefix.replace('\n', ' ').replace('\t', ' ')

        # Print the final result without the prefix
        print(f"Final Result {i+1}:")
        print(result_no_prefix)
        print("**********************************************************")

        final_line += '\t' + result_no_prefix.strip() 

    # Write the final line to the output file
    #output_file.write(final_line + '\n')
    if val_indx % print_at == 0:
        print(final_line)
        print(f"Processed {val_indx} images. Elapsed time: {time.time() - start_time:.2f} seconds")

#output_file.close() 

print(f"Processing complete! Total time: {time.time() - start_time:.2f} seconds")