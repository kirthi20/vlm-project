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

def prepare_image_safely(image, target_size=224):
    """
    Prepare image with consistent sizing while preserving aspect ratio
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a square canvas with the target size
    canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))  # Black background
    
    # Calculate scaling to fit image within target size while preserving aspect ratio
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Center the image on the canvas
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas.paste(image, (x_offset, y_offset))
    
    return canvas

def preprocess_function(examples):
    batch_size = len(examples["images"])
    
    all_chosen_input_ids = []
    all_chosen_attention_masks = []
    all_chosen_pixel_values = []
    all_rejected_input_ids = []
    all_rejected_attention_masks = []
    all_rejected_pixel_values = []
    
    for i in range(batch_size):
        # Get single example
        image = prepare_image_safely(examples["images"][i][0])
        
        # Combine prompt and responses into full conversations
        chosen_messages = examples["prompt"][i] + examples["chosen"][i]
        rejected_messages = examples["prompt"][i] + examples["rejected"][i]
        
        # Apply chat template to get properly formatted text
        chosen_text = processor.apply_chat_template(chosen_messages, tokenize=False)
        rejected_text = processor.apply_chat_template(rejected_messages, tokenize=False)
        
        # Process with a fixed max length to ensure consistency
        chosen_inputs = processor(
            text=chosen_text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=4096  # Use your desired max length
        )
        
        rejected_inputs = processor(
            text=rejected_text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=4096  # Use your desired max length
        )
        
        # Collect results - all tensors should now be the same length
        all_chosen_input_ids.append(chosen_inputs["input_ids"])
        all_chosen_attention_masks.append(chosen_inputs["attention_mask"])
        all_chosen_pixel_values.append(chosen_inputs["pixel_values"])
        all_rejected_input_ids.append(rejected_inputs["input_ids"])
        all_rejected_attention_masks.append(rejected_inputs["attention_mask"])
        all_rejected_pixel_values.append(rejected_inputs["pixel_values"])
    
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    try:
        # Pad all sequences and create batches
        padded_chosen_input_ids = pad_sequence(all_chosen_input_ids, batch_first=True, padding_value=pad_token_id)
        padded_chosen_attention_masks = pad_sequence(all_chosen_attention_masks, batch_first=True, padding_value=0)
        padded_rejected_input_ids = pad_sequence(all_rejected_input_ids, batch_first=True, padding_value=pad_token_id)
        padded_rejected_attention_masks = pad_sequence(all_rejected_attention_masks, batch_first=True, padding_value=0)
    except RuntimeError as e:
        print(f"Error during padding: {e}")
        print("Debugging tensor shapes and dtypes:")
        
        # Check for inconsistent dtypes or devices
        for i, seq in enumerate(all_chosen_input_ids):
            print(f"Chosen input_ids {i}: shape={seq.shape}, dtype={seq.dtype}, device={seq.device}")
        for i, seq in enumerate(all_rejected_input_ids):
            print(f"Rejected input_ids {i}: shape={seq.shape}, dtype={seq.dtype}, device={seq.device}")
        
        raise
    
    return {
        "input_ids_chosen": padded_chosen_input_ids,
        "attention_mask_chosen": padded_chosen_attention_masks,
        "pixel_values_chosen": torch.cat(all_chosen_pixel_values, dim=0),
        "input_ids_rejected": padded_rejected_input_ids,
        "attention_mask_rejected": padded_rejected_attention_masks,
        "pixel_values_rejected": torch.cat(all_rejected_pixel_values, dim=0),
    }

# Preprocess dataset
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=dataset.column_names
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