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
import time
from datetime import datetime
import random

# Configuration - MODIFY THESE AS NEEDED
# USE_MULTI_GPU = False  # Set to True for multi-GPU training
# GPU_IDS = [3]  # For single GPU, use [3]. For multi-GPU, use [2, 3] or [0, 1] etc.

# Initialize wandb (optional)
wandb.init(project="smolvlm-qlora-dpo-finetuning")

# Setup device configuration
# if USE_MULTI_GPU and len(GPU_IDS) > 1:
#     device_map = "auto"  # Automatic distribution
#     device = torch.device(f"cuda:{GPU_IDS[0]}")  # Primary device
# else:
#     gpu_id = GPU_IDS[0]
#     torch.cuda.set_device(gpu_id)  # Directly set the GPU
#     device = torch.device(f"cuda:{gpu_id}")
#     device_map = {"": gpu_id}  # Map to specific GPU

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # only GPU 3 is visible

torch.cuda.set_device(0)  # GPU 3 is now referred to as cuda:0
device = torch.device("cuda:0")
device_map = {"": 0}  # or device_map={"": torch.cuda.current_device()}

# Start timing
start_time = time.time()
start_datetime = datetime.now()
print(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# QLoRA configuration - 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    int8_threshold=6.0,  # Optional: threshold for outlier detection
    llm_int8_threshold=6.0,  # Optional: specific threshold for LLMs
    llm_int8_has_fp16_weight=False,  # Optional: keep some weights in fp16
)

# Load model and processor with quantization
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# FIX 2: Use current device for device mapping
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,  # Use current device instead of hardcoded
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
#model = get_peft_model(model, peft_config)
# Replace the get_peft_model section (lines 88-89) with:
# Check if model already has peft config
if hasattr(model, 'peft_config'):
    print("Model already has PEFT config, skipping LoRA application")
else:
    model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Add these imports at the top
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image


# FIX 4: Add label_ids_chosen and label_ids_rejected to prevent the warning
def preprocess_for_dpo(example):
    # Convert image mode + resize
    image = example["images"]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))  # Avoid loading large image blobs

    # Apply chat template
    chosen_text = processor.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
    rejected_text = processor.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)

    return {
        "image": image,  # Just return PIL object
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
    }

from torch.utils.data import IterableDataset

class DPOIterableDataset(IterableDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __iter__(self):
        for example in self.dataset:
            image = example["image"]
            chosen_text = example["chosen_text"]
            rejected_text = example["rejected_text"]

            # Use processor on the fly (in-memory)
            chosen = self.processor(
                text=chosen_text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=4096,
            )

            rejected = self.processor(
                text=rejected_text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=4096,
            )

            # Create label tensors with padding masked
            label_ids_chosen = chosen["input_ids"].squeeze(0).clone()
            label_ids_rejected = rejected["input_ids"].squeeze(0).clone()
            
            # Mask padding tokens in labels
            label_ids_chosen[label_ids_chosen == self.processor.tokenizer.pad_token_id] = -100
            label_ids_rejected[label_ids_rejected == self.processor.tokenizer.pad_token_id] = -100

            # Clean up the PIL image after processing
            image.close()
            del image
            
            # Also delete the texts to free memory
            del chosen_text
            del rejected_text
            del example

            yield {
                "input_ids_chosen": chosen["input_ids"].squeeze(0),
                "attention_mask_chosen": chosen["attention_mask"].squeeze(0),
                "pixel_values_chosen": chosen["pixel_values"].squeeze(0),
                "input_ids_rejected": rejected["input_ids"].squeeze(0),
                "attention_mask_rejected": rejected["attention_mask"].squeeze(0),
                "pixel_values_rejected": rejected["pixel_values"].squeeze(0),
                "label_ids_chosen": chosen["input_ids"].squeeze(0),
                "label_ids_rejected": rejected["input_ids"].squeeze(0),
            }

def data_collator(examples):

    # Periodically run garbage collection
    if random.random() < 0.1:  # 10% of the time
        gc.collect()

    return {
        key: torch.stack([ex[key] for ex in examples])
        for key in examples[0]
    }

streaming_dataset = load_dataset(
    "HuggingFaceH4/rlaif-v_formatted",
    split="train",
    streaming=True
).take(2000)  # Limit to 2000 samples for experiment

# Load dataset
#dataset = load_dataset("HuggingFaceH4/rlaif-v_formatted", split="train", streaming=True).take(2000)
#eval_dataset = load_dataset("HuggingFaceH4/rlaif-v_formatted", split="test", streaming=True).take(500)

dataset_size = 2000
batch_size = 32  # adjust to your actual batch size
max_steps = dataset_size // batch_size
save_steps = 20  
logging_steps = 100
eval_steps = save_steps 

processed_stream = streaming_dataset.map(
    preprocess_for_dpo,
    batched=False,
)

torch_dataset = DPOIterableDataset(processed_stream, processor)

# DPO training configuration optimized for QLoRA
training_args = DPOConfig(
    output_dir="./modelcache/smolvlm-qlora-dpo-finetuned",
    # huggingface reference params
    num_train_epochs=5, 
    per_device_train_batch_size=1,  # Can use larger batch size with QLoRA
    gradient_accumulation_steps=32,  # Reduced due to larger batch size
    save_steps=save_steps,
    save_strategy="steps",
    save_total_limit=1,
    gradient_checkpointing=True,
    logging_steps=logging_steps,
    bf16=True,  # Use bf16 instead of fp16 for better stability
    report_to="wandb",
    dataloader_num_workers=8,  # Parallel data loading
    torch_compile=False,  # FIX 5: Disable torch.compile for quantized models
    # eval_steps=save_steps,  # Less frequent than the reference's 10
    # eval_strategy="steps",
    # per_device_eval_batch_size=1,
    max_steps=max_steps,  # Set max steps based on dataset size
    # Other items
    # ddp_find_unused_parameters=False if USE_MULTI_GPU else None, # Support for multi-GPU
    max_grad_norm=0.3,
    remove_unused_columns=False,  # FIX 6: Keep all columns for DPO
)

# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=torch_dataset,
    # eval_dataset=eval_dataset,  # Evaluation dataset
    data_collator=data_collator,  # This should handle tokenization
    processing_class=processor,  # Use processor's tokenizer
    peft_config=peft_config,
    ref_model=None,
)

# Start training
trainer.train()

# Save the fine-tuned adapter
trainer.save_model("./modelcache/smolvlm-qlora-dpo-final")

print("Training completed!")

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Cleanup memory after training
cleanup_memory()

# End timing
end_time = time.time()
end_datetime = datetime.now()
total_time = end_time - start_time

# Calculate time metrics
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("-" * 50)
print(f"Training completed at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {hours}h {minutes}m {seconds}s")
print(f"Time per sample: {total_time/dataset_size:.2f} seconds")
print(f"Samples per hour: {dataset_size/(total_time/3600):.0f}")

# Extrapolate for your full dataset
if dataset_size < 80000:
    estimated_80k_time = (80000 / dataset_size) * total_time
    est_hours = int(estimated_80k_time // 3600)
    est_minutes = int((estimated_80k_time % 3600) // 60)
    print(f"Estimated time for 80k samples: {est_hours}h {est_minutes}m")
