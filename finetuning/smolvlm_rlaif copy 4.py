import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb
import os
from datetime import datetime
import time

os.environ["HF_HOME"] = "/data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

# Initialize wandb
wandb.init(project="smolvlm-qlora-dpo-finetuning", mode="online")

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(0)  # GPU 3 is now referred to as cuda:0
device = torch.device("cuda:0")
device_map = {"": 0}  # or device_map={"": torch.cuda.current_device()}

# Start timing
start_time = time.time()
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Model configuration
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

# 8-bit quantization config (more stable than 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,  # Higher r for QLoRA to compensate for quantization
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "v_proj", 
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    init_lora_weights="gaussian",
    use_dora=True,  # Enable DORA for better performance
)

# Apply LoRA BEFORE creating the trainer
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Optional: see how many parameters are trainable

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
        image = examples["images"][i][0]
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
        
        processed["prompt"].append(prompt[0]["content"])
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
    #remove_columns=dataset.column_names
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
    max_steps=5000,  # Limit steps for 80k samples
    report_to="wandb",
    ddp_find_unused_parameters=False,
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