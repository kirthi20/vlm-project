import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers.image_utils import load_image
from datasets import load_dataset
from PIL import Image
import pandas as pd
import io

# Set device
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on GPU
model = SentenceTransformer('clip-ViT-L-14', device=device)

# Load COCO Karpathy validation dataset
dataset = load_dataset("yerevann/coco-karpathy", split="validation")

def analyze_coco_validation(batch_size=64):
    results = []
    
    # Process entire dataset in batches more efficiently
    for i in range(0, len(dataset), batch_size):
        # Get batch slice
        batch = dataset[i:i+batch_size]
        
        # Load images - check what fields are actually available
        batch_images = []
        batch_ids = []
        
        for item in batch:
            # Use the actual image field (not URL)
            image = item['image']  # This should be the PIL image directly
            batch_images.append(image)
            batch_ids.append(item['cocoid'])
        
        # Single encoding call for text (move outside loop for efficiency)
        if i == 0:  # Only encode text once
            text_embeddings = model.encode(['male', 'female'], convert_to_tensor=True, device=device)
        
        # Encode images
        img_embeddings = model.encode(batch_images, convert_to_tensor=True, device=device)
        
        # Calculate similarities
        similarities = util.cos_sim(img_embeddings, text_embeddings)
        
        # Process results
        for j, (sim, coco_id) in enumerate(zip(similarities, batch_ids)):
            sim_cpu = sim.cpu()
            
            results.append({
                'coco_id': coco_id,
                'male_similarity': sim_cpu[0].item(),
                'female_similarity': sim_cpu[1].item(),
                'male_prob': F.softmax(sim_cpu, dim=0)[0].item(),
                'female_prob': F.softmax(sim_cpu, dim=0)[1].item(),
                'bias_score': sim_cpu[0].item() - sim_cpu[1].item(),
                'weat_score': (sim_cpu[0].item() - sim_cpu[1].item()) / (sim_cpu[0].item() + sim_cpu[1].item())
            })
    
    return results

# Run analysis
print(f"Processing {len(dataset)} validation images...")
results = analyze_coco_validation()
pd.DataFrame(results).to_csv('coco_bias_results.csv', index=False)
print("Results saved to coco_bias_results.csv")