import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers.image_utils import load_image
from datasets import load_dataset
from PIL import Image
import pandas as pd
import io

# Set device
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on GPU
model = SentenceTransformer('clip-ViT-L-14', device=device)

# Load COCO Karpathy validation dataset
dataset = load_dataset("yerevann/coco-karpathy", split="validation")

def analyze_coco_validation(batch_size=64):
    results = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        # Convert images from bytes to PIL
        batch_images = []
        batch_ids = []
        
        for j, item in enumerate(batch):
            image_url = item['url']
            image = load_image(image_url)
            batch_images.append(image)
            batch_ids.append(item['cocoid'])  # COCO image ID
        
        # Encode batch on GPU
        img_embeddings = model.encode(batch_images, convert_to_tensor=True, device=device)
        text_embeddings = model.encode(['male', 'female'], convert_to_tensor=True, device=device)
        
        # Calculate similarities on GPU
        similarities = util.cos_sim(img_embeddings, text_embeddings)
        
        for j, sim in enumerate(similarities):
            sim_cpu = sim.cpu()
            
            results.append({
                'coco_id': batch_ids[j],
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