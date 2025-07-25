import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
import wandb
from datetime import datetime
import time
from PIL import Image
import gc

# Initialize wandb
wandb.init(project="smolvlm-rlaif-m1", mode="online")

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
torch.cuda.set_device(0)
device = torch.device("cuda:0")
device_map = "auto"

# Start timing
start_time = time.time()
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Model configuration
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Load processor and model
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_id, revision="main")

# Set image size to match model expectations
processor.image_processor.size = {"longest_edge": 384}  # Reduced from 512
processor.image_processor.max_image_size = {"longest_edge": 384}

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    revision="main",
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Use bfloat16 from the start
)

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    init_lora_weights="gaussian",
    use_dora=True,
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def prepare_dataset(example):
    """Prepare dataset in DPO format with proper image handling"""
    try:
        # Handle image
        image = example.get("image")
        if image is None:
            return None
            
        # Ensure RGB without resizing (let processor handle it)
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        
        # Format for DPO
        formatted_example = {
            "prompt": "<image>" + example["question"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "images": [image]  # Keep as list
        }
        
        return formatted_example
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

# Load dataset with streaming to avoid memory issues
print("Loading dataset...")
dataset = load_dataset(
    "openbmb/RLAIF-V-Dataset",
    split="train",
    streaming=True  # Enable streaming
)

# For testing, limit to smaller subset
# dataset = dataset.take(1000)  # Uncomment for testing

# Apply preprocessing with error handling
dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names
).filter(lambda x: x is not None)  # Remove failed examples

# Convert streaming dataset to regular dataset for DPOTrainer
# This will process examples on-the-fly
print("Preparing dataset for training...")
train_dataset = dataset.take(30000)  # Adjust number as needed

# Training configuration
training_args = DPOConfig(
    output_dir="./smolvlm-dpo-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Reduced batch size
    gradient_accumulation_steps=4,   # Effective batch size = 16
    gradient_checkpointing=True,     # Enable to save memory
    optim="adamw_torch_fused",      # Fused optimizer for speed
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    tf32=True,
    dataloader_num_workers=4,  # Reduced workers
    remove_unused_columns=False,
    max_grad_norm=1.0,
    report_to="wandb",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,  # Disable if causing issues
    # Add these for better data loading
    dataloader_prefetch_factor=2,
    dataloader_persistent_workers=True,
)

# Initialize trainer
print("Initializing trainer...")
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # No ref model for LoRA
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor,
)

# Clear cache before training
gc.collect()
torch.cuda.empty_cache()

# Train
print("Starting training...")
try:
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nTraining error: {e}")
    raise

# Save model
print("Saving model...")
trainer.save_model("./smolvlm-rlaif-m1")
processor.save_pretrained("./smolvlm-rlaif-m1")

# Training summary
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("-" * 50)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {hours}h {minutes}m {seconds}s")
print(f"Model saved to ./smolvlm-rlaif-m1")

# Final cleanup
gc.collect()
torch.cuda.empty_cache()
wandb.finish()