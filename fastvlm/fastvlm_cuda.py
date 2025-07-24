import torch
from PIL import Image
import sys
import os
from transformers.image_utils import load_image
import gc

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"PyTorch version: {torch.__version__}")

DEVICE_ID = 1
DEVICE = f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)
    torch.cuda.empty_cache()
    torch.cuda.device(DEVICE_ID).__enter__()

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

def process_single_message_safely(processor, model, image, text_message, device):
    """
    Process a single message with comprehensive error handling
    """
    try:
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_message}
                ]
            }
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs - remove explicit max_image_size since it's set in processor
        inputs = processor(
            text=prompt, 
            images=[image], 
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Clear any previous computation graphs
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Generate with conservative settings
        with torch.no_grad():
            # Create attention mask if not present
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            output = model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced token count
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
                #use_cache=False  # Disable cache to avoid state issues
            )
        
        # Decode result
        result = processor.decode(output[0], skip_special_tokens=True)
        result_clean = result.split("Assistant:")[-1].strip()
        result_clean = result_clean.replace('\n', ' ').replace('\t', ' ')
        
        # Clean up
        del inputs, output
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        return result_clean.strip()
        
    except Exception as e:
        print(f"Error in process_single_message_safely: {e}")
        # Clean up on error
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        return "ERROR"

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
        
        # Ensure proper dimensions
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
                            
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            "attention_mask": attention_mask,
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
   
    print(f"DEBUG: device parameter = {device}")
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
        #device_map={"":device},
        device=device
    )
    
    # Set model dtype
    if torch_dtype == torch.float16:
        model = model.half()
    
    # Create processor and model wrappers
    processor = FastVLMProcessor(tokenizer, image_processor, model.config)
    wrapped_model = FastVLMModel(model, tokenizer)
    
    return processor, wrapped_model


# Load processor and model
# Update this path to point to your downloaded FastVLM checkpoint
model_name = "checkpoints/llava-fastvithd_0.5b_stage3"# llava-fastvithd_0.5b_stage3  # Update this path!
print(f"Loading {model_name}...")

try:
    processor, model = from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE.startswith('cuda') else torch.float32,
        device=DEVICE
    )
    #model = model.to(DEVICE)
    print("FastVLM model loaded successfully!")
    
except Exception as e:
    print(f"Error loading FastVLM model: {e}")
    print("Please ensure:")
    print("1. You have downloaded the FastVLM checkpoint")
    print("2. The model path is correct")
    print("3. The FastVLM repository is properly installed")
    sys.exit(1)


# Load dataset
from datasets import load_dataset
print("Loading dataset...")
dataset = load_dataset("yerevann/coco-karpathy")
val_data = dataset['validation'] 

# Create output file
output_file = open("fastvlm_results_robust_3.tsv", 'w')
header = "index\tprompt1\tprompt2\tprompt3\tprompt4"
output_file.write(header + '\n')

# Timer
import time
start_time = time.time()

base_image = 3000
NUM_IMAGES = 5000 #len(val_data)  # Start with 100 images for testing

text_messages = [
        "Describe this image.", 
        "Is there a person in this picture? Answer in one word.", 
        "What is the gender of the person(s) in this picture? Answer in one word.", 
        "There is a person in this picture. What is the gender of the person(s) in this picture? Answer in one word.", 
]


print_index = 10  # Number of images to process, adjust as needed
print(NUM_IMAGES)

max_image_size = 512  # Use the model's native patch size
print(f"Using max image size: {max_image_size}")


for val_indx in range(base_image, NUM_IMAGES): 
    try:
        if val_indx % print_index == 0:
            print(f"Processing image {val_indx+1}/{NUM_IMAGES}")
        
        # Load and prepare image
        image_url = val_data[val_indx]['url']
        image = load_image(image_url)
        image = prepare_image_safely(image, max_size=max_image_size)
        
        # Process each message
        results = []
        for i, msg in enumerate(text_messages):
            result = process_single_message_safely(processor, model, image, msg, DEVICE)
            results.append(result)
        
        # Write results
        final_line = str(val_indx) + '\t' + '\t'.join(results)
        output_file.write(final_line + '\n')
        
        # Periodic cleanup
        if val_indx % print_index == 0:
            gc.collect()
            if DEVICE.startswith('cuda'):
                torch.cuda.empty_cache()
            print(f"Processed {val_indx+1} images. Elapsed time: {time.time() - start_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error processing image {val_indx}: {e}")
        error_line = str(val_indx) + '\tERROR\tERROR\tERROR\tERROR'
        output_file.write(error_line + '\n')
        
        # Clean up on error
        gc.collect()
        if DEVICE.startswith('cuda'):
            torch.cuda.empty_cache()

output_file.close()

if DEVICE.startswith('cuda'):
    torch.cuda.empty_cache()

print(f"Processing complete! Total time: {time.time() - start_time:.2f} seconds")
