import torch
from PIL import Image
import sys
import os
from transformers.image_utils import load_image
import gc
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import time
from tqdm.auto import tqdm

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
from fastvlm_cuda import prepare_image_safely, process_single_message_safely, FastVLMProcessor, FastVLMModel, from_pretrained


class COCOEvalDataset(Dataset):
    """Custom dataset for COCO evaluation with FastVLM"""
    def __init__(self, max_samples=None):
        self.dataset = load_dataset("yerevann/coco-karpathy")
        self.val_data = self.dataset['validation']
        self.max_samples = max_samples if max_samples else len(self.val_data)
        
        self.prompts = [
            "Describe this image.", 
            "Is there a person in this picture? Answer in one word.", 
            "What is the gender of the person(s) in this picture? Answer in one word.", 
            "There is a person in this picture. What is the gender of the person(s) in this picture? Answer in one word.", 
        ]
    
    def __len__(self):
        return min(self.max_samples, len(self.val_data))
    
    def __getitem__(self, idx):
        return {
            'index': idx,
            'image_url': self.val_data[idx]['url'],
            'prompts': self.prompts
        }


def collate_fn(batch):
    """Custom collate function to handle our data structure"""
    return batch


def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Configuration
    model_name = "checkpoints/llava-fastvithd_0.5b_stage3"
    total_images = 5000
    batch_size = 1  # Process one image at a time (you can increase this if your GPU memory allows)
    max_image_size = 512
    
    # Only print from the main process
    if accelerator.is_main_process:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Local process index: {accelerator.local_process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Loading {model_name}...")
    
    # Start timer
    start_time = time.time()
    
    # Load model and processor
    # Each process loads its own copy of the model
    processor, model = from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device=accelerator.device
    )
    
    if accelerator.is_main_process:
        print("Model loaded successfully!")
    
    # Create dataset and dataloader
    dataset = COCOEvalDataset(max_samples=total_images)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,  # Important: don't shuffle for consistent results
        collate_fn=collate_fn
    )
    
    # Prepare dataloader with accelerator
    # Note: We don't prepare the model since it's already on the correct device
    dataloader = accelerator.prepare(dataloader)
    
    # Storage for results
    all_results = []
    
    # Process images
    progress_bar = tqdm(
        total=len(dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Processing on GPU {accelerator.local_process_index}"
    )
    
    for batch in dataloader:
        batch_results = []
        
        for item in batch:
            try:
                # Load and prepare image
                image = load_image(item['image_url'])
                image = prepare_image_safely(image, max_size=max_image_size)
                
                # Process each prompt
                results = []
                for prompt in item['prompts']:
                    result = process_single_message_safely(
                        processor, 
                        model, 
                        image, 
                        prompt, 
                        accelerator.device
                    )
                    results.append(result)
                
                batch_results.append({
                    'index': item['index'],
                    'results': results
                })
                
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error processing image {item['index']}: {e}")
                batch_results.append({
                    'index': item['index'],
                    'results': ['ERROR'] * len(item['prompts'])
                })
            
            # Periodic cleanup
            if item['index'] % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        all_results.extend(batch_results)
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Gather results from all processes
    if accelerator.is_main_process:
        print("Gathering results from all processes...")
    
    # Use gather_object to collect results from all processes
    all_results_gathered = gather_object(all_results)
    
    # Only the main process writes the output
    if accelerator.is_main_process:
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        print("Writing results to file...")
        
        # Sort results by index to maintain order
        all_results_gathered.sort(key=lambda x: x['index'])
        
        # Write results
        with open("fastvlm_results_accelerate.tsv", 'w') as f:
            # Write header
            header = "index\tprompt1\tprompt2\tprompt3\tprompt4"
            f.write(header + '\n')
            
            # Write data
            for result in all_results_gathered:
                line = str(result['index']) + '\t' + '\t'.join(result['results'])
                f.write(line + '\n')
        
        print("Results saved to fastvlm_results_accelerate.tsv")
        print(f"Processed {len(all_results_gathered)} images in total")
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()