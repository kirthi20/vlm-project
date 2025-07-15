import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb
from datetime import datetime
import time
from PIL import Image

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
processor = AutoProcessor.from_pretrained(model_id, revision="main")

processor.image_processor.size = {"longest_edge": 512}
processor.image_processor.max_image_size = {"longest_edge": 512}

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    revision="main",
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

def ensure_rgb(example):
    # Convert the image to RGB if it's not already
    image = example["images"][0]
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        #image = image.resize((512, 512))
        example["images"] = [image]

    return example

# Load dataset with streaming
print("Loading dataset...")
train_dataset = load_dataset(
    "HuggingFaceH4/rlaif-v_formatted",
    split="train"
).take(1000)

# Apply preprocessing
train_dataset = train_dataset.map(ensure_rgb, num_proc=32)

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
    train_dataset=train_dataset,
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