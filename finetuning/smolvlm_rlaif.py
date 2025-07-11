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

# Initialize wandb (optional)
wandb.init(project="smolvlm-qlora-dpo-finetuning")

# Set device
DEVICE_ID = 2
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


def prepare_image_safely(image, max_size=224):
    """
    Prepare image with very conservative sizing
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    
    # Scale down to a safe size
    scale = min(max_size / width, max_size / height)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Make dimensions divisible by 8 (often helps with vision models)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Ensure minimum size
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        #print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image

def preprocess_function(examples):
    images = examples["images"]
    images = [prepare_image_safely(img[0]) for img in images]

    #questions = examples["question"]
    # chosen_responses = examples["chosen"]
    # rejected_responses = examples["rejected"]
    chosen_texts = examples["chosen"]
    rejected_texts = examples["rejected"]
    
    # chosen_texts = []
    # rejected_texts = []
    
    # for question, chosen_response, rejected_response in zip(questions, chosen_responses, rejected_responses):
    #     # Try using chat template if available
    #     try:
    #         chosen_messages = [
    #             {"role": "user", "content": question},
    #             {"role": "assistant", "content": chosen_response}
    #         ]
    #         rejected_messages = [
    #             {"role": "user", "content": question},
    #             {"role": "assistant", "content": rejected_response}
    #         ]
            
    #         chosen_text = processor.tokenizer.apply_chat_template(
    #             chosen_messages, tokenize=False, add_generation_prompt=True
    #         )
    #         rejected_text = processor.tokenizer.apply_chat_template(
    #             rejected_messages, tokenize=False, add_generation_prompt=True
    #         )
    #     except Exception as e:
    #         print(f"Error applying chat template: {e}")
    #         print("Falling back to simple format.")
    #         # Fallback to simple format
    #         chosen_text = f"Question: {question}\nAnswer: {chosen_response}"
    #         rejected_text = f"Question: {question}\nAnswer: {rejected_response}"
        
    #     chosen_texts.append(chosen_text)
    #     rejected_texts.append(rejected_text)
    
    # Process with the formatted texts
    chosen_inputs = processor(
        text=chosen_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    
    rejected_inputs = processor(
        text=rejected_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
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
    remove_columns=dataset["train"].column_names,
    cache_file_names=None,  # Disable caching
    load_from_cache_file=False
)

# DPO training configuration optimized for QLoRA
training_args = DPOConfig(
    output_dir="./smolvlm-qlora-dpo-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Can use larger batch size with QLoRA
    gradient_accumulation_steps=4,  # Reduced due to larger batch size
    gradient_checkpointing=True,
    learning_rate=2e-4,  # Higher LR often works better with QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
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
)

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
trainer.save_model("./smolvlm-qlora-dpo-final")

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