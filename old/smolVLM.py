import os
import argparse
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from tqdm import tqdm
import pickle
from collections import defaultdict
from old.CHAIREvaluator import CHAIREvaluator

class SmolVLMEvaluator:
    """Class for generating captions using SmolVLM model and evaluating them with CHAIR"""
    
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct", device=None):
        """
        Initialize the SmolVLM model for generating captions
        
        Args:
            model_name: Hugging Face model name/path
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading SmolVLM model {model_name} on {self.device}...")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
        ).to(self.device)
        
        print("Model loaded successfully")
    
    def generate_caption(self, image_path, prompt="Describe this image in detail:"):
        """
        Generate a caption for an image
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to guide the captioning
            
        Returns:
            Generated caption text
        """
        # Load and preprocess the image
        image = load_image(image_path)
        
        # Create messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=150)
        
        # Decode the caption
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # Extract the assistant's response (after the prompt)
        if "assistant" in generated_text.lower():
            response_start = generated_text.lower().find("assistant")
            if response_start != -1:
                # Find the actual text after "assistant"
                text_start = generated_text.find(":", response_start)
                if text_start != -1:
                    generated_text = generated_text[text_start + 1:].strip()
        
        return generated_text
    
    def generate_captions_for_dataset(self, image_dir, image_ids, prompt="Describe this image in detail:", batch_size=1):
        """
        Generate captions for a dataset of images
        
        Args:
            image_dir: Directory containing images
            image_ids: List of image IDs to process
            prompt: Text prompt for captioning
            batch_size: Batch size for processing
            
        Returns:
            List of generated captions and corresponding image IDs
        """
        captions = []
        processed_ids = []
        
        for i, img_id in enumerate(tqdm(image_ids, desc="Generating captions")):
            try:
                # Construct image path (assuming COCO format)
                img_path = os.path.join(image_dir, f"COCO_val2014_{img_id:012d}.jpg")
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} not found, skipping")
                    continue
                
                # Generate caption
                caption = self.generate_caption(img_path, prompt)
                captions.append(caption)
                processed_ids.append(img_id)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_ids)} images")
                    
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
        
        return captions, processed_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM on the CHAIR metric using MSCOCO dataset")
    parser.add_argument("--coco_path", type=str, required=True, help="Path to MSCOCO dataset directory")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolVLM-Instruct", 
                       help="HuggingFace model name/path for SmolVLM")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to evaluate (set to -1 for all)")
    parser.add_argument("--device", type=str, default=None, help="Device to run model on (cuda/cpu)")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail:",
                        help="Prompt for captioning")
    parser.add_argument("--cache_chair", type=str, default=None, 
                       help="Path to cached CHAIR evaluator (to speed up loading)")
    parser.add_argument("--save_cache", action="store_true", 
                       help="Save CHAIR evaluator cache for future use")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CHAIR evaluator (from cache if available)
    if args.cache_chair and os.path.exists(args.cache_chair):
        print(f"Loading CHAIR evaluator from cache: {args.cache_chair}")
        with open(args.cache_chair, 'rb') as f:
            chair_evaluator = pickle.load(f)
    else:
        print(f"Initializing CHAIR evaluator from MSCOCO at {args.coco_path}")
        chair_evaluator = CHAIREvaluator(args.coco_path)
        
        # Save cache if requested
        if args.save_cache:
            cache_path = args.cache_chair or os.path.join(args.output_dir, "chair_evaluator.pkl")
            print(f"Saving CHAIR evaluator cache to {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(chair_evaluator, f)
    
    # Get list of image IDs to evaluate
    with open(os.path.join(args.coco_path, 'annotations/instances_val2014.json'), 'r') as f:
        instances = json.load(f)
    
    all_image_ids = [img['id'] for img in instances['images']]
    
    # Limit the number of images if specified
    if args.num_images > 0:
        image_ids = all_image_ids[:args.num_images]
    else:
        image_ids = all_image_ids
    
    print(f"Will evaluate on {len(image_ids)} images")
    
    # Initialize SmolVLM evaluator
    smolvlm_evaluator = SmolVLMEvaluator(args.model_name, args.device)
    
    # Generate captions for the dataset
    image_dir = os.path.join(args.coco_path, 'val2014')
    captions, processed_ids = smolvlm_evaluator.generate_captions_for_dataset(
        image_dir, image_ids, args.prompt
    )
    
    # Save generated captions
    captions_output = {
        'model': args.model_name,
        'prompt': args.prompt,
        'captions': [{'image_id': img_id, 'caption': cap} for img_id, cap in zip(processed_ids, captions)]
    }
    
    captions_path = os.path.join(args.output_dir, 'generated_captions.json')
    with open(captions_path, 'w') as f:
        json.dump(captions_output, f, indent=2)
    
    print(f"Saved generated captions to {captions_path}")
    
    # Evaluate with CHAIR metric
    chair_results = chair_evaluator.evaluate_captions(captions, processed_ids)
    
    # Save CHAIR results
    results_path = os.path.join(args.output_dir, 'chair_results.json')
    with open(results_path, 'w') as f:
        json.dump(chair_results, f, indent=2)
    
    # Print summary of results
    print("\n===== CHAIR Evaluation Results =====")
    print(f"Model: {args.model_name}")
    print(f"Evaluated on {len(processed_ids)} images")
    print(f"CHAIRs: {chair_results['overall_metrics']['CHAIRs']:.4f}")
    print(f"CHAIRi: {chair_results['overall_metrics']['CHAIRi']:.4f}")
    print(f"Detailed results saved to {results_path}")


if __name__ == "__main__":
    main()
