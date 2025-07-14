import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
import os
from PIL import Image
from transformers.image_utils import load_image
import gc
from torch.nn.utils.rnn import pad_sequence

# Initialize wandb (optional)
wandb.init(project="smolvlm-qlora-dpo-finetuning")

# Set device
DEVICE_ID = 3
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# QLoRA configuration - 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model and processor with quantization
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": DEVICE_ID},  # Map entire model to cuda:0
    trust_remote_code=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA for QLoRA fine-tuning
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

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("HuggingFaceH4/rlaif-v_formatted", split="train") # openbmb/RLAIF-V-Dataset

# Add these imports at the top
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

# Replace the prepare_image_safely function with this GPU-accelerated version
# def prepare_image_safely_batch(images, target_size=224):
#     """
#     Prepare images batch-wise on GPU with consistent sizing while preserving aspect ratio
#     """
#     processed_images = []
    
#     # Define transforms that can run on GPU
#     transform = transforms.Compose([
#         transforms.Resize((target_size, target_size)),  # This will distort, but it's faster
#         transforms.ConvertImageDtype(torch.float32),
#     ])
    
#     for image in images:
#         # Convert to RGB if needed
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to tensor and move to GPU
#         image_tensor = to_tensor(image).to(device)
        
#         # Apply transforms on GPU
#         image_tensor = transform(image_tensor)
        
#         # Convert back to PIL for processor (unfortunately necessary)
#         processed_image = to_pil_image(image_tensor.cpu())
#         processed_images.append(processed_image)
    
#     return processed_images

def prepare_image_safely_batch(images, target_size=224):
    """
    Prepare images on CPU only - no GPU operations
    """
    processed_images = []
    
    for image in images:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((target_size, target_size))
        processed_images.append(image)
    
    return processed_images

def preprocess_function(examples):
    batch_size = len(examples["images"])
    
    # Process all images at once
    images = [img[0] for img in examples["images"]]
    processed_images = prepare_image_safely_batch(images)
    
    # Pre-compile all texts first (faster than doing it in the loop)
    chosen_texts = []
    rejected_texts = []
    
    for i in range(batch_size):
        chosen_messages = examples["prompt"][i] + examples["chosen"][i]
        rejected_messages = examples["prompt"][i] + examples["rejected"][i]
        
        chosen_texts.append(processor.apply_chat_template(chosen_messages, tokenize=False))
        rejected_texts.append(processor.apply_chat_template(rejected_messages, tokenize=False))
    
    # Process all chosen examples at once
    chosen_inputs = processor(
        text=chosen_texts,
        images=processed_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    )
    
    # Process all rejected examples at once
    rejected_inputs = processor(
        text=rejected_texts,
        images=processed_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    )
    
    return {
        "input_ids_chosen": chosen_inputs["input_ids"],
        "attention_mask_chosen": chosen_inputs["attention_mask"],
        "pixel_values_chosen": chosen_inputs["pixel_values"],
        "input_ids_rejected": rejected_inputs["input_ids"],
        "attention_mask_rejected": rejected_inputs["attention_mask"],
        "pixel_values_rejected": rejected_inputs["pixel_values"],
    }

# Preprocess dataset
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=dataset.column_names,
    num_proc=4,  # Use multiple processes for faster preprocessing
    cache_file_name=None, 
    load_from_cache_file=False
)

processed_dataset.save_to_disk("./datacache/processed_rlaif_dataset")
# processed_dataset = load_from_disk("datacache/processed_rlaif_dataset")

# Print a sample to verify preprocessing
print(processed_dataset["input_ids_chosen"][0])
print(processed_dataset["attention_mask_chosen"][0])
print(processed_dataset["pixel_values_chosen"][0].shape)
print(processed_dataset["input_ids_rejected"][0])
print(processed_dataset["attention_mask_rejected"][0])
print(processed_dataset["pixel_values_rejected"][0].shape)

# DPO training configuration optimized for QLoRA
training_args = DPOConfig(
    output_dir="./modelcache/smolvlm-qlora-dpo-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Can use larger batch size with QLoRA
    gradient_accumulation_steps=4,  # Reduced due to larger batch size
    gradient_checkpointing=True,
    learning_rate=2e-4,  # Higher LR often works better with QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_steps=1000,
    evaluation_strategy="no",
    beta=0.1,  # DPO beta parameter
    loss_type="sigmoid",  # DPO loss type
    bf16=True,  # Use bf16 instead of fp16 for better stability
    optim="paged_adamw_32bit",  # Memory-efficient optimizer
    max_grad_norm=0.3,  # Gradient clipping for stability
    push_to_hub=False,
    report_to="wandb",
    dataloader_pin_memory=True,  # Pin memory for faster data transfer
    dataloader_num_workers=4,  # Parallel data loading
    save_only_model=True,  # Don't save optimizer states
    save_total_limit=2,    # Keep only last 2 checkpoints
    torch_compile=True, # Enable torch.compile for performance
    dataloader_prefetch_factor=2,  # Prefetch more batches
)

# Compile the model for faster execution (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")

# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    tokenizer=processor,
    peft_config=peft_config,
    ref_model=None,  # No reference model needed for DPO
)

# Start training
trainer.train()

# Save the fine-tuned adapter
trainer.save_model("./modelcache/smolvlm-qlora-dpo-final")

# Merge and save the full model (optional - requires more memory)
# from peft import PeftModel
# base_model = AutoModelForVision2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# model = PeftModel.from_pretrained(base_model, "./smolvlm-qlora-dpo-final")
# model = model.merge_and_unload()
# model.save_pretrained("./smolvlm-qlora-merged")

# Inference example
# def generate_response(image, prompt):
#     inputs = processor(text=prompt, images=image, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )
    
#     response = processor.decode(outputs[0], skip_special_tokens=True)
#     return response

print("Training completed!")

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Cleanup memory after training
cleanup_memory()


# Memory usage comparison:
# Regular LoRA: ~8-12GB VRAM for 256M model
# QLoRA: ~4-6GB VRAM for 256M model
