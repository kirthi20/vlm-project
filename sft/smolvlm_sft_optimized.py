import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from PIL import Image
import random
import io
import wandb
from transformers.image_utils import load_image

# Initialize wandb
wandb.init(project="smolvlm-m1-sft", mode="online")

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)  # GPU 3 is now referred to as cuda:0
device = torch.device("cuda:0")
device_map = {"": 0}  # or device_map={"": torch.cuda.current_device()}

class COCOCaptionDataset(Dataset):
    def __init__(self, dataset, processor, num_captions=3):
        self.dataset = dataset
        self.processor = processor
        self.num_captions = num_captions
        
        # Define instruction templates
        self.instruction_templates = [
            "Describe this image in detail.",
            "What can you see in this image?",
            "Provide a detailed caption for this image.",
            "What is shown in this picture?",
            "Generate a description of this image.",
            "Explain what this image contains.",
            "Write a caption for this image.",
            "What does this image depict?"
        ]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load image
        image = item['image']
        
        # Get captions and randomly select specified number
        captions = item['sentences']
        selected_captions = random.sample(captions, min(self.num_captions, len(captions)))
        
        # Randomly select one caption for this training example
        caption = random.choice(selected_captions)
        
        # Randomly select an instruction template
        instruction = random.choice(self.instruction_templates)
        
        # Format as instruction-following prompt - NO TOKENIZATION HERE
        text_input = f"<|user|>\n<image>{instruction}\n<|end|>\n<|assistant|>\n{caption}\n<|end|>"
        
        # Return raw data - let the data collator handle tokenization
        return {
            "images": image,
            "text": text_input
        }

class VisionSeq2SeqDataCollator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch):
        # Extract images and texts
        images = [item["images"] for item in batch]
        texts = [item["text"] for item in batch]
        
        # Batch tokenization - much faster!
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Set labels for training (mask instruction, only predict response)
        labels = inputs["input_ids"].clone()
        
        # Process each sequence in the batch to mask instructions
        for i, text in enumerate(texts):
            # Find where assistant response starts for this sequence
            tokens = self.processor.tokenizer.encode(text, add_special_tokens=False)
            assistant_token = self.processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]
            
            try:
                assistant_idx = tokens.index(assistant_token)
                # Mask everything up to and including the assistant token
                labels[i, :assistant_idx + 1] = -100
            except ValueError:
                # If we can't find assistant token, mask first half as fallback
                labels[i, :len(tokens)//2] = -100
        
        inputs["labels"] = labels
        return inputs

def ensure_rgb(example):
    # Convert the image to RGB if it's not already
    image_url = example['url']
    image = load_image(image_url)

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    example["image"] = image
    return example

def main():
    # Model and dataset paths
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    dataset_name = "yerevann/coco-karpathy"
    
    # Load processor and model
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_name, revision="main")

    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        revision="main",
        #quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Set up padding token if not exists
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")

    # Use subset for multiple epochs approach (recommended based on your overfitting)
    print("Using subset of data for multiple epochs...")
    dataset = dataset.shuffle(seed=42).select(range(20000))  # 20k samples instead of full dataset

    dataset = dataset.map(ensure_rgb, num_proc=8)
    
    # Create training dataset
    train_dataset = COCOCaptionDataset(dataset, processor, num_captions=3)
    
    # Use custom data collator for batch tokenization
    data_collator = VisionSeq2SeqDataCollator(processor, max_length=512)
    
    # Training arguments - adjusted for subset + multiple epochs
    training_args = TrainingArguments(
        output_dir="./smolvlm-m1-sft",
        num_train_epochs=3,  # More epochs since using subset
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=300,  # Reduced since fewer total steps
        learning_rate=3e-5,  # Slightly higher for subset approach
        weight_decay=0.1,  # Increased for better regularization
        logging_steps=50,
        save_steps=500,  # More frequent saves
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model("./smolvlm-m1-sft")
    processor.save_pretrained("./smolvlm-m1-sft")
    
    print("Training completed!")

if __name__ == "__main__":
    main()