import torch
from PIL import Image
import sys
import os
from transformers.image_utils import load_image
import gc
import multiprocessing as mp

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
from datasets import load_dataset


from fastvlm_cuda import prepare_image_safely, process_single_message_safely, FastVLMProcessor, FastVLMModel, from_pretrained

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"PyTorch version: {torch.__version__}")

# Load processor and model
# Update this path to point to your downloaded FastVLM checkpoint
model_name = "checkpoints/llava-fastvithd_0.5b_stage3"# llava-fastvithd_0.5b_stage3  # Update this path!
print(f"Loading {model_name}...")

# Timer
import time
start_time = time.time()

def worker_process(worker_id, gpu_id, start_idx, end_idx, model_name, output_file_path):
    """Worker process that runs on a specific GPU"""
    
    # Set up device for this worker
    device = f"cuda:{gpu_id}"
    print(f"Worker {worker_id} with {gpu_id}: Starting on device {device}, processing images {start_idx} to {end_idx-1}")
    
    # Set CUDA device
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    print_index = 125  # Number of images to process, adjust as needed
    max_image_size = 512  # Use the model's native patch size
    
    try:
        # Load model on this GPU
        print(f"Worker {worker_id} with {gpu_id}: Loading model...")
        processor, model = from_pretrained(  
            model_name,
            torch_dtype=torch.float16,
            device=device
        )
        print(f"Worker {worker_id} with {gpu_id}: Model loaded successfully!")
        
        # Load dataset
        print(f"Worker {worker_id} with {gpu_id}: Loading dataset...")
        dataset = load_dataset("yerevann/coco-karpathy")
        val_data = dataset['validation']
        
        # Create output file for this worker
        with open(output_file_path, 'w') as output_file:
            header = "index\tprompt1\tprompt2\tprompt3\tprompt4"
            output_file.write(header + '\n')
            
            # Process assigned range of images
            for val_indx in range(start_idx, end_idx):
                try:
                    if val_indx % print_index == 0:
                        print(f"Worker {worker_id} with {gpu_id}: Processing image {val_indx}/{end_idx-1}")
                    
                    # Load and prepare image
                    image_url = val_data[val_indx]['url']
                    image = load_image(image_url)
                    image = prepare_image_safely(image, max_size=max_image_size)

                    text_messages = [
                            "Describe this image.", 
                            "Is there a person in this picture? Answer in one word.", 
                            "What is the gender of the person(s) in this picture? Answer in one word.", 
                            "There is a person in this picture. What is the gender of the person(s) in this picture? Answer in one word.", 
                    ]
                    
                    # Process each message
                    results = []
                    for msg in text_messages:
                        result = process_single_message_safely(processor, model, image, msg, device)
                        results.append(result)
                    
                    # Write results
                    final_line = str(val_indx) + '\t' + '\t'.join(results)
                    output_file.write(final_line + '\n')
                    
                    # Periodic cleanup
                    if val_indx % print_index == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Worker {worker_id} with {gpu_id}: Error processing image {val_indx}: {e}")
                    error_line = str(val_indx) + '\tERROR\tERROR\tERROR\tERROR'
                    output_file.write(error_line + '\n')
                    
                    # Clean up on error
                    gc.collect()
                    torch.cuda.empty_cache()
        
        print(f"Worker {worker_id} with {gpu_id}: Completed processing {end_idx - start_idx} images")
        
    except Exception as e:
        print(f"Worker {worker_id} with {gpu_id}: Fatal error: {e}")
    finally:
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

def main():
    """Main function to orchestrate parallel processing"""
    
    # Configuration
    model_name = "checkpoints/llava-fastvithd_0.5b_stage3"
    total_images = 5000
    gpu_ids = [0, 1, 2, 3]  # Use GPUs 0 and 1
    # Calculate work distribution
    images_per_worker = total_images // len(gpu_ids)
    
    print(f"Total images: {total_images}")
    print(f"Images per worker: {images_per_worker}")
    print(f"Using GPUs: {gpu_ids}")
    
    # Create work assignments
    work_assignments = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * images_per_worker
        end_idx = start_idx + images_per_worker
        output_file = f"fastvlm_results_worker_{i}.tsv"
        work_assignments.append((i, gpu_id, start_idx, end_idx, model_name, output_file))
        print(f"Worker {i}: GPU {gpu_id}, images {start_idx}-{end_idx-1}, output: {output_file}")
    
    # Start timer
    start_time = time.time()
    
    # Create and start worker processes
    processes = []
    for assignment in work_assignments:
        p = mp.Process(target=worker_process, args=assignment)
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"Processing complete! Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Set start method for multiprocessing (important for CUDA)
    mp.set_start_method('spawn', force=True)
    main()