import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
import wandb
import os
from datetime import datetime
import time

# Initialize wandb
wandb.init(project="smolvlm-qlora-dpo-finetuning", mode="online")

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0")

# Start timing
start_time = time.time()
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Model configuration
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

# 8-bit quantization config (more stable than 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_compile=False,
    trust_remote_code=True
)

# # LoRA configuration
# peft_config = LoraConfig(
#     r=16,  # Slightly higher for better performance
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )

# Simple preprocessing function
def preprocess_batch(examples):
    """Process a batch of examples for DPO training"""
    processed = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "images": []
    }
    
    for i in range(len(examples["prompt"])):
        # Get image
        image = examples["images"][i]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224)) 
        
        # Build conversations
        prompt = examples["prompt"][i]
        chosen = prompt + examples["chosen"][i]
        rejected = prompt + examples["rejected"][i]
        
        # Apply chat template
        chosen_text = processor.apply_chat_template(chosen, tokenize=False)
        rejected_text = processor.apply_chat_template(rejected, tokenize=False)
        
        processed["prompt"].append(prompt[0]["content"] if isinstance(prompt, list) else "")
        processed["chosen"].append(chosen_text)
        processed["rejected"].append(rejected_text)
        processed["images"].append(image)
    
    return processed

# Load dataset with streaming
print("Loading dataset...")
dataset = load_dataset(
    "HuggingFaceH4/rlaif-v_formatted",
    split="train",
    streaming=True
)

# Apply preprocessing
dataset = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=4,  # Process in small batches
    remove_columns=dataset.column_names
)

# Training configuration
training_args = DPOConfig(
    output_dir="./smolvlm-dpo-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16
    gradient_checkpointing=True,
    optim="adamw_8bit",  # 8-bit optimizer to save memory
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    tf32=True,  # Enable TF32 for faster training
    dataloader_num_workers=4,
    remove_unused_columns=False,
    label_names=["labels"],
    max_steps=5000,  # Limit steps for 80k samples
    report_to="wandb",
    ddp_find_unused_parameters=False,
    group_by_length=True,  # Group similar length sequences
    length_column_name="length",
    dataloader_prefetch_factor=2,
    dataloader_persistent_workers=True,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # No ref model needed for QLoRA
    args=training_args,
    train_dataset=dataset,
    #tokenizer=processor,
    processing_class=processor,
    #peft_config=peft_config,
)

# Train
print("Starting training...")
trainer.train()

# Save model
trainer.save_model("./smolvlm-dpo-final")
processor.save_pretrained("./smolvlm-dpo-final")

# Training summary
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("-" * 50)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {hours}h {minutes}m {seconds}s")
print("Model saved to ./smolvlm-dpo-final")