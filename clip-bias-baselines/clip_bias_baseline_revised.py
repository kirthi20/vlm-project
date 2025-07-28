import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from PIL import Image
import pandas as pd
import io
from transformers.image_utils import load_image

# Set device
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on GPU
model = SentenceTransformer('clip-ViT-L-14', device=device)

# Load COCO Karpathy validation dataset
dataset = load_dataset("yerevann/coco-karpathy", split="validation")

# Define descriptor groups
male_text = ['male', 'man', 'boy', 'masculine', 'he']
female_text = ['female', 'woman', 'girl', 'feminine', 'she']

# Encode text prompts
male_embeddings = model.encode(male_text, convert_to_tensor=True, device=device)
female_embeddings = model.encode(female_text, convert_to_tensor=True, device=device)

def analyze_coco_validation(batch_size=64):
    results = []

    for i in range(0, len(dataset), batch_size):
        batch_images = []
        batch_ids = []

        # Load and preprocess image batch
        for j in range(i, min(i + batch_size, len(dataset))):
            item = dataset[j]
            try:
                image_url = item['url']
                image = load_image(image_url)
                batch_images.append(image)
                batch_ids.append(item['cocoid'])
            except Exception as e:
                print(f"Error loading image {j}: {e}")
                continue

        if not batch_images:
            continue

        # Encode image batch
        img_embeddings = model.encode(batch_images, convert_to_tensor=True, device=device)

        # Compute cosine similarities
        male_sim = util.cos_sim(img_embeddings, male_embeddings)  # [batch_size, len(male_text)]
        female_sim = util.cos_sim(img_embeddings, female_embeddings)  # [batch_size, len(female_text)]

        # Average scores
        avg_male_sim = male_sim.mean(dim=1)  # [batch_size]
        avg_female_sim = female_sim.mean(dim=1)  # [batch_size]

        for coco_id, male_score, female_score in zip(batch_ids, avg_male_sim, avg_female_sim):
            results.append({
                'index': i,
                'coco_id': coco_id,
                'avg_male_similarity': male_score.item(),
                'avg_female_similarity': female_score.item(),
                'difference': male_score.item() - female_score.item()
            })

    return results

# Run analysis
print(f"Processing {len(dataset)} validation images...")
results = analyze_coco_validation()

# Convert to DataFrame and save
df = pd.DataFrame(results)
df.to_csv('coco_bias_results.csv', index=False)

# Report average difference
mean_diff = df['difference'].mean()
print(f"Average CLIP similarity difference (male - female): {mean_diff:.4f}")
print("Results saved to all_clip_scores.csv")
