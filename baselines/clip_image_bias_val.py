import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from PIL import Image
import pandas as pd
import io
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Set device
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on GPU
model = SentenceTransformer('clip-ViT-L-14', device=device)

# Load COCO Karpathy validation dataset
dataset = load_dataset("yerevann/coco-karpathy", split="validation")

def analyze_coco_validation(batch_size=64):
    results = []
    
    # Encode text once
    text_embeddings = model.encode(['male', 'female'], convert_to_tensor=True, device=device)
    
    for i in range(0, len(dataset), batch_size):
        batch_images = []
        batch_ids = []
        
        # Process individual items
        for j in range(i, min(i + batch_size, len(dataset))):
            item = dataset[j]
            try:
                image_url = item['url']  # or whatever the URL field is called
                image = load_image(image_url)
                batch_images.append(image)
                batch_ids.append(item['cocoid'])
            except Exception as e:
                print(f"Error loading image {j}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Encode images
        img_embeddings = model.encode(batch_images, convert_to_tensor=True, device=device)
        
        # Calculate similarities
        similarities = util.cos_sim(img_embeddings, text_embeddings)
        
        # Process results
        for sim, coco_id in zip(similarities, batch_ids):
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