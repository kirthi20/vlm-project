import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"

import torch
from PIL import Image
from datetime import datetime
import wandb
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor

# Import FastVLM utilities
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# Set up environment
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize wandb
wandb.init(project="fastvlm-qlora-dpo-finetuning", mode="online")

# GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

class SimpleFastVLMProcessor:
    """Simplified processor for DPO training"""
    
    def __init__(self, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        """Apply chat template for conversation formatting"""
        conv = conv_templates["plain"].copy()
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                text_parts = []
                has_image = False
                
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                        elif item["type"] == "image":
                            has_image = True
                else:
                    text_parts.append(content)
                
                text = " ".join(text_parts)
                
                if has_image:
                    if self.model_config.mm_use_im_start_end:
                        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
                    else:
                        text = DEFAULT_IMAGE_TOKEN + '\n' + text
                
                conv.append_message(conv.roles[0], text)
                    
            elif role == "assistant":
                if isinstance(content, list):
                    content = " ".join([item["text"] if isinstance(item, dict) else item for item in content])
                conv.append_message(conv.roles[1], content)
        
        prompt = conv.get_prompt()
        if add_generation_prompt:
            prompt += f"{conv.roles[1]}: "
        
        return prompt
    
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """Process inputs for training"""
        if isinstance(text, list):
            # Batch processing
            results = []
            for i, single_text in enumerate(text):
                single_image = images[i] if images and i < len(images) else None
                result = self._process_single(single_text, single_image, return_tensors)
                results.append(result)
            
            if return_tensors == "pt":
                # Collate batch
                return self._collate_batch(results)
            else:
                # Return first item for DPO
                return results[0]
        else:
            return self._process_single(text, images, return_tensors)
    
    def _process_single(self, text, images, return_tensors):
        """Process a single text-image pair"""
        # Process images
        image_tensor = None
        if images:
            if not isinstance(images, list):
                images = [images]
            
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                processed_images.append(img)
            
            image_tensor = process_images(processed_images, self.image_processor, self.model_config)
            if image_tensor is not None:
                image_tensor = image_tensor.to(device)
        
        # Tokenize text
        if return_tensors == "pt":
            input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            
            return {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "images": image_tensor
            }
        else:
            # For DPO tokenization phase
            input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(input_ids, int):
                input_ids = [input_ids]
            
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "images": image_tensor
            }
    
    def _collate_batch(self, batch):
        """Collate a batch of processed items"""
        max_length = max(item["input_ids"].shape[-1] for item in batch)
        
        input_ids_list = []
        attention_mask_list = []
        images_list = []
        
        for item in batch:
            # Pad sequences
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_length = max_length - ids.shape[-1]
            
            if pad_length > 0:
                ids = torch.cat([ids, torch.full((1, pad_length), self.pad_token_id, device=device)], dim=1)
                mask = torch.cat([mask, torch.zeros(1, pad_length, device=device)], dim=1)
            
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            
            if item["images"] is not None:
                images_list.append(item["images"])
        
        return {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
            "images": torch.cat(images_list, dim=0) if images_list else None
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode tokens"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def save_pretrained(self, path):
        """Save processor components"""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.image_processor.save_pretrained(path)


def ensure_rgb_and_resize(example):
    """Preprocess images in the dataset"""
    if "images" in example and example["images"]:
        processed_images = []
        for img in example["images"]:
            if isinstance(img, Image.Image):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                processed_images.append(img)
        example["images"] = processed_images
    return example


# Load model using simplified approach
model_path = "checkpoints/llava-fastvithd_0.5b_stage3"
model_id = model_path.split("/")[-1]

print(f"Loading FastVLM model from {model_path}...")

try:
    # Load model components directly
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    
    # Create processor
    processor = SimpleFastVLMProcessor(tokenizer, image_processor, model.config)
    
    # Set generation config
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    print("FastVLM model loaded successfully!")
    
except Exception as e:
    print(f"Error loading FastVLM model: {e}")
    import sys
    sys.exit(1)

# Apply LoRA for finetuning
peft_config = LoraConfig(
    r=8,
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
    use_dora=True,
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Enable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Load and preprocess dataset
print("Loading dataset...")
train_dataset = load_dataset(
    "HuggingFaceH4/rlaif-v_formatted",
    split="train"
).take(100)

# Apply preprocessing
train_dataset = train_dataset.map(ensure_rgb_and_resize, num_proc=4)

# Training configuration
training_args = DPOConfig(
    output_dir="./fastvlm-dpo-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_torch",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    tf32=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    report_to="wandb",
    ddp_find_unused_parameters=False,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor,
)

# Start training
print("Starting training...")
import time
start_time = time.time()

try:
    trainer.train()
    
    # Save model
    trainer.save_model("./fastvlm-dpo-final")
    processor.save_pretrained("./fastvlm-dpo-final")
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
    
    # Save checkpoint even if training fails
    trainer.save_model("./fastvlm-dpo-checkpoint")
    processor.save_pretrained("./fastvlm-dpo-checkpoint")

finally:
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Training summary
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("-" * 50)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
    print("Model saved to ./fastvlm-dpo-final")
    
    wandb.finish()
