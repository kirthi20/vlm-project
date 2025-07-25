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
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Get captions and randomly select specified number
        captions = item['sentences']
        selected_captions = random.sample(captions, min(self.num_captions, len(captions)))
        
        # Randomly select one caption for this training example
        caption = random.choice(selected_captions)['raw']
        
        # Randomly select an instruction template
        instruction = random.choice(self.instruction_templates)
        
        # Format as instruction-following prompt
        text_input = f"<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{caption}\n<|end|>"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=text_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Set labels for training (mask instruction, only predict response)
        labels = inputs["input_ids"].clone()
        
        # Find where assistant response starts
        assistant_start = None
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        for i, token in enumerate(tokens):
            if "<|assistant|>" in str(token):
                assistant_start = i + 1
                break
        
        # Mask everything before assistant response
        if assistant_start is not None:
            labels[:assistant_start] = -100
        
        inputs["labels"] = labels
        
        return inputs

def main():
    # Model and dataset paths
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    dataset_name = "yerevann/coco-karpathy"
    
    # Load processor and model
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Set up padding token if not exists
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    
    # Create training dataset
    train_dataset = COCOCaptionDataset(dataset, processor, num_captions=3)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./smolvlm-coco-sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        report_to="tensorboard",
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
    trainer.save_model("./smolvlm-coco-sft-final")
    processor.save_pretrained("./smolvlm-coco-sft-final")
    
    print("Training completed!")

if __name__ == "__main__":
    main()