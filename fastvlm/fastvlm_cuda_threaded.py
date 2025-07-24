import torch
from PIL import Image
import sys
import os
from transformers.image_utils import load_image
import gc
import multiprocessing as mp
from datasets import load_dataset
import time

# IMPORTANT: Set multiprocessing start method FIRST, before any CUDA operations
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

def worker_process(worker_id, gpu_id, start_idx, end_idx, model_name, output_file_path, fastvlm_path):
    """Worker process that runs on a specific GPU"""
    
    # Add the FastVLM repo to Python path INSIDE the worker
    sys.path.append(fastvlm_path)
    
    # Import FastVLM/CUDA components INSIDE the worker process
    # This ensures each process initializes CUDA independently
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from fastvlm_cuda import prepare_image_safely, process_single_message_safely, FastVLMProcessor, FastVLMModel, from_pretrained
    
    # Set up device for this worker
    device = f"cuda:{gpu_id}"
    print(f"Worker {worker_id} on GPU {gpu_id}: Starting on device {device}, processing images {start_idx} to {end_idx-1}")
    
    # Set CUDA device for this process
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    
    # Verify we're on the correct GPU
    print(f"Worker {worker_id}: Current CUDA device = {torch.cuda.current_device()}")

    print_index = 125  # Number of images to process, adjust as needed
    max_image_size = 512  # Use the model's native patch size
    
    try:
        # Load model on this GPU
        print(f"Worker {worker_id} on GPU {gpu_id}: Loading model...")
        processor, model = from_pretrained(  
            model_name,
            torch_dtype=torch.float16,
            device=device
        )
        print(f"Worker {worker_id} on GPU {gpu_id}: Model loaded successfully!")
        
        # Load dataset
        print(f"Worker {worker_id} on GPU {gpu_id}: Loading dataset...")
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
                        print(f"Worker {worker_id} on GPU {gpu_id}: Processing image {val_indx}/{end_idx-1}")
                    
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
                    print(f"Worker {worker_id} on GPU {gpu_id}: Error processing image {val_indx}: {e}")
                    error_line = str(val_indx) + '\tERROR\tERROR\tERROR\tERROR'
                    output_file.write(error_line + '\n')
                    
                    # Clean up on error
                    gc.collect()
                    torch.cuda.empty_cache()
        
        print(f"Worker {worker_id} on GPU {gpu_id}: Completed processing {end_idx - start_idx} images")
        
    except Exception as e:
        print(f"Worker {worker_id} on GPU {gpu_id}: Fatal error: {e}")
    finally:
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

def main():
    """Main function to orchestrate parallel processing"""
    
    print(f"Main process: CUDA available: {torch.cuda.is_available()}")
    print(f"Main process: CUDA device count: {torch.cuda.device_count()}")
    print(f"Main process: PyTorch version: {torch.__version__}")
    
    # Configuration
    fastvlm_path = "./ml-fastvlm"  # Change this to your actual FastVLM repo path
    model_name = "checkpoints/llava-fastvithd_0.5b_stage3"
    total_images = 5000
    gpu_ids = [0, 1, 2, 3]  # Use GPUs 0, 1, 2, and 3
    
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
        # Handle remainder for last worker
        if i == len(gpu_ids) - 1:
            end_idx = total_images
        output_file = f"fastvlm_results_worker_{i}.tsv"
        work_assignments.append((i, gpu_id, start_idx, end_idx, model_name, output_file, fastvlm_path))
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
    
    # # Optionally, merge all worker output files
    # print("\nMerging output files...")
    # with open("fastvlm_results_merged.tsv", 'w') as merged_file:
    #     # Write header once
    #     merged_file.write("index\tprompt1\tprompt2\tprompt3\tprompt4\n")
        
    #     # Merge all worker files
    #     for i in range(len(gpu_ids)):
    #         worker_file = f"fastvlm_results_worker_{i}.tsv"
    #         try:
    #             with open(worker_file, 'r') as f:
    #                 lines = f.readlines()[1:]  # Skip header
    #                 merged_file.writelines(lines)
    #             print(f"Merged {worker_file}")
    #         except Exception as e:
    #             print(f"Error merging {worker_file}: {e}")
    
    # print("All results merged into fastvlm_results_merged.tsv")

if __name__ == "__main__":
    main()