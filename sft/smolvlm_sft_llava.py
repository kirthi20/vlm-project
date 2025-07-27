import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb
from datetime import datetime
import time
from PIL import Image

# Initialize wandb
wandb.init(project="smolvlm-sft-llava-finetuning", mode="online")

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
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct" #"HuggingFaceTB/SmolVLM-500M-Instruct" 

# 8-bit quantization config (more stable than 4-bit)
#bnb_config = BitsAndBytesConfig(
#    load_in_8bit=True
#)

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, revision="main")

processor.image_processor.size = {"longest_edge": 512}
processor.image_processor.max_image_size = {"longest_edge": 512}

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    revision="main",
    #quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True
)

#model = prepare_model_for_kbit_training(model)

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
        
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        example["images"] = image

    return example

def prepare_sft_format(example):
    """
    Convert the llava-instruct-mix-vsft format to SFTTrainer format.
    Expected format has 'messages' field with conversation format.
    """
    # The dataset has 'messages' field with conversation format
    # We need to extract the prompt and response
    messages = example["messages"]
    
    # Find the user message (usually the first one) and assistant response
    user_message = None
    assistant_message = None
    
    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
        elif message["role"] == "assistant":
            assistant_message = message["content"]
    
    # Format for SFT: combine user prompt with <image> token and assistant response
    if user_message and assistant_message:
        # Add image token to the beginning of user message
        example["text"] = f"{user_message}\n{assistant_message}"
        #example["images"] = example["image"]
    
    return example

# Load dataset and randomly sample 20k examples
print("Loading dataset...")
full_dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train")

# Randomly sample 20k examples with seed
train_dataset = full_dataset.shuffle(seed=42).select(range(100))#20000

print(f"Dataset size: {len(train_dataset)} examples")

# Apply preprocessing
train_dataset = train_dataset.map(ensure_rgb, num_proc=16)
train_dataset = train_dataset.map(prepare_sft_format, num_proc=16)

# Remove unused columns, keep only what SFTTrainer needs
columns_to_keep = ['text', 'images']
columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]
train_dataset = train_dataset.remove_columns(columns_to_remove)

print("Sample preprocessed example:")
print(f"Text: {train_dataset[0]['text'][:200]}...")
print(f"Image type: {type(train_dataset[0]['images'])}")

# Training configuration
training_args = SFTConfig(
    output_dir="./smolvlm-sft-llava-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    gradient_checkpointing=False,
    optim="adamw_torch",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    tf32=True,  # Enable TF32 for faster training
    dataloader_num_workers=8,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    report_to="wandb",
    ddp_find_unused_parameters=False,
    # SFT specific settings
    max_seq_length=2048,  # Maximum sequence length
    dataset_text_field="text",  # Field containing the text data
)

# Set pad_token for the processor before creating trainer
#if not hasattr(processor, 'pad_token') or processor.pad_token is None:
#    processor.pad_token = processor.tokenizer.eos_token

# Add missing tokenizer methods to processor
#processor.convert_tokens_to_ids = processor.tokenizer.convert_tokens_to_ids
#processor.eos_token = processor.tokenizer.eos_token

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor.tokenizer,
    # For vision models, we need to specify how to handle images
    #dataset_kwargs={
    #    "skip_prepare_dataset": False,
    #}
)

# Train
print("Starting training...")
trainer.train()

# Save model
trainer.save_model("./smolvlm-sft-llava-finetuned")
processor.save_pretrained("./smolvlm-sft-llava-finetuned")

# Training summary
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("-" * 50)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {hours}h {minutes}m {seconds}s")
print("Model saved to ./smolvlm-sft-llava-finetuned")
