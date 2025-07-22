import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"

import sys
import torch
from PIL import Image
from datetime import datetime
import wandb
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from transformers.image_utils import load_image

# Set up environment
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize wandb
wandb.init(project="fastvlm-qlora-dpo-finetuning", mode="online")

# GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Add FastVLM to path
fastvlm_path = "./ml-fastvlm"
sys.path.append(fastvlm_path)

# Import FastVLM components
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

class SimpleFastVLMProcessor:
    """Simplified processor for DPO training"""
    
    def __init__(self, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Add required tokenizer attributes for DPO compatibility
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        self.vocab_size = len(tokenizer)
        
        # Add special tokens
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.pad_token = tokenizer.pad_token
        self.unk_token = getattr(tokenizer, 'unk_token', None)
    
    def encode(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        """Encode text using the tokenizer"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)
    
    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """Decode tokens"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.tokenizer)
        
    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        """Apply chat template for conversation formatting"""
        # Use a simple format if messages are already formatted strings
        if isinstance(messages, str):
            return messages
            
        # Handle list of messages
        if not isinstance(messages, list):
            messages = [messages]
            
        # Try to use a simple conversation format first
        try:
            conv = conv_templates.get("vicuna_v1", conv_templates.get("plain", None))
            if conv is None:
                # Fallback to manual formatting
                formatted_messages = []
                for message in messages:
                    if isinstance(message, dict):
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        if isinstance(content, list):
                            # Handle multimodal content
                            text_parts = []
                            has_image = False
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                    elif item.get("type") == "image":
                                        has_image = True
                            content = " ".join(text_parts)
                            if has_image and hasattr(self, 'model_config'):
                                content = DEFAULT_IMAGE_TOKEN + '\n' + content
                        formatted_messages.append(f"{role}: {content}")
                
                prompt = "\n".join(formatted_messages)
                if add_generation_prompt:
                    prompt += "\nassistant: "
                return prompt
            
            # Use conversation template
            conv = conv.copy()
            
            # Ensure separators are strings
            if conv.sep is None:
                conv.sep = "\n"
            if conv.sep2 is None:
                conv.sep2 = conv.sep
            
            for message in messages:
                if not isinstance(message, dict):
                    continue
                    
                role = message.get("role", "")
                content = message.get("content", "")
                
                if isinstance(content, list):
                    text_parts = []
                    has_image = False
                    
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                has_image = True
                        else:
                            text_parts.append(str(item) if item is not None else "")
                    
                    content = " ".join(filter(None, text_parts))
                    
                    if has_image and hasattr(self, 'model_config') and self.model_config.mm_use_im_start_end:
                        content = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + content
                    elif has_image:
                        content = DEFAULT_IMAGE_TOKEN + '\n' + content
                
                # Ensure content is a string
                content = str(content) if content is not None else ""
                
                # Map roles
                if role.lower() in ["user", "human"]:
                    conv.append_message(conv.roles[0], content)
                elif role.lower() in ["assistant", "gpt", "model"]:
                    conv.append_message(conv.roles[1], content)
            
            prompt = conv.get_prompt()
            
            if add_generation_prompt and len(conv.roles) > 1:
                # Add the assistant prompt starter
                if not prompt.endswith(conv.roles[1] + ": "):
                    prompt += conv.sep + conv.roles[1] + ": "
            
            return prompt
            
        except Exception as e:
            print(f"Error in apply_chat_template: {e}")
            # Final fallback - just concatenate messages
            formatted = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join([str(item.get("text", "") if isinstance(item, dict) else item) for item in content])
                    formatted.append(f"{role}: {content}")
            prompt = "\n".join(formatted)
            if add_generation_prompt:
                prompt += "\nassistant: "
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
                    img = load_image(img)
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

            # Create labels with -100 for image tokens
            labels = input_ids.clone()
            labels[labels == IMAGE_TOKEN_INDEX] = -100  # Ignore index for loss
            
            return {
                "input_ids": input_ids.to(device),
                "labels": labels.to(device),
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
            
            # Add bounds checking for token IDs
            # input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
            # if isinstance(input_ids, torch.Tensor):
            #     input_ids = input_ids.tolist()
            # if isinstance(input_ids, int):
            #     input_ids = [input_ids]
            
            # Clamp token IDs to valid range
            #labels = [token_id if token_id != IMAGE_TOKEN_INDEX else -100 for token_id in input_ids]

            processed_ids = []
            vocab_size = len(self.tokenizer)
            for token_id in input_ids:
                if token_id == IMAGE_TOKEN_INDEX:
                    processed_ids.append(token_id)  # Keep -200
                elif token_id < 0:
                    processed_ids.append(0)  # Replace other negative values
                elif token_id >= vocab_size:
                    processed_ids.append(vocab_size - 1)  # Cap at vocab size
                else:
                    processed_ids.append(token_id)

            return {
                "input_ids": input_ids,
                # "labels": labels,
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
    
    def save_pretrained(self, path):
        """Save processor components"""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        # Save image processor config if available
        if hasattr(self.image_processor, 'save_pretrained'):
            try:
                self.image_processor.save_pretrained(path)
            except:
                pass  # Some custom processors might not support this


def ensure_rgb_and_resize(example):
    """Preprocess images in the dataset"""
    # Handle both individual examples and batched examples
    if "images" in example and example["images"]:
        images = example["images"]
        # Check if it's a batched call
        if isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], list):  # Batched
                processed_batch = []
                for img_list in images:
                    processed_images = []
                    for img in img_list:
                        if isinstance(img, Image.Image):
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img = img.resize((512, 512), Image.Resampling.LANCZOS)
                            processed_images.append(img)
                    processed_batch.append(processed_images)
                example["images"] = processed_batch
            else:  # Single example
                processed_images = []
                for img in images:
                    if isinstance(img, Image.Image):
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        processed_images.append(img)
                example["images"] = processed_images
    return example

# Create a wrapper that handles -200 tokens
class FastVLMEmbeddingWrapper(torch.nn.Module): # ADDED FOR VAL 3
    def __init__(self, embed_layer, pad_token_id):
        super().__init__()
        self.embed_layer = embed_layer
        self.pad_token_id = pad_token_id
        
    def forward(self, input_ids):
        # Replace -200 with pad token for embedding lookup
        mask = input_ids == IMAGE_TOKEN_INDEX
        safe_input_ids = input_ids.clone()
        safe_input_ids[mask] = self.pad_token_id
        
        # Get embeddings
        embeddings = self.embed_layer(safe_input_ids)
        
        # Zero out embeddings for image positions (they'll be filled by vision encoder)
        embeddings[mask] = 0
        
        return embeddings

def combine_prompt_with_response(example): # added for val 4
    """Combine prompt with chosen/rejected to create full conversations"""
    # The prompt already has the image reference
    prompt_messages = example['prompt']
    
    # Combine prompt + chosen
    example['chosen'] = prompt_messages + example['chosen']
    
    # Combine prompt + rejected  
    example['rejected'] = prompt_messages + example['rejected']
    
    return example

# Load FastVLM model using original method
model_path = "checkpoints/llava-fastvithd_0.5b_stage3"
print(f"Loading FastVLM model from {model_path}...")

try:
    # Disable torch init (FastVLM requirement)
    disable_torch_init()
    
    # Get model name
    model_name = get_model_name_from_path(model_path)
    
    # Load pretrained model with FastVLM's method
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    
    # Create processor
    processor = SimpleFastVLMProcessor(tokenizer, image_processor, model.config)

    if hasattr(model.get_model(), 'embed_tokens'):
        original_embed = model.get_model().embed_tokens
        model.get_model().embed_tokens = FastVLMEmbeddingWrapper(original_embed, tokenizer.pad_token_id)

    # Move model to device and set to fp16
    model = model.to(device)
    # if device.type == 'cuda': # COMMENTED FOR VAL 4
    #     model = model.half()
    
    print("FastVLM model loaded successfully!")
    
except Exception as e:
    print(f"Error loading FastVLM model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Apply LoRA for finetuning
peft_config = LoraConfig(
    r=16, # VAL <= 3: 8
    lora_alpha=32,  # VAL <= 3: 8
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

# Debug: Print dataset structure
print("Dataset columns:", train_dataset.column_names)
print("First example keys:", train_dataset[0].keys())
if "chosen" in train_dataset[0]:
    print("Chosen format:", train_dataset[0]["chosen"][:200] if isinstance(train_dataset[0]["chosen"], str) else train_dataset[0]["chosen"])
if "rejected" in train_dataset[0]:
    print("Rejected format:", train_dataset[0]["rejected"][:200] if isinstance(train_dataset[0]["rejected"], str) else train_dataset[0]["rejected"])

# Apply preprocessing
train_dataset = train_dataset.map(ensure_rgb_and_resize, num_proc=16)
train_dataset = train_dataset.map(combine_prompt_with_response, num_proc=16)

# Training configuration
training_args = DPOConfig(
    output_dir="./fastvlm-dpo-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4, # VAL 1 = 2, 2 and 3 = 4
    gradient_accumulation_steps=2, # VAL 1 = 8, 2 and 3 = 4, VAL 4 = 2
    gradient_checkpointing=True,
    optim="adamw_torch",
    learning_rate=5e-6, # VAL 1 = 5e-5, 2 and 3 = 2e-5, VAL 4 = 5e-6 (smaller rate to support larger batch size)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=False, #torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=False, #torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    tf32=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    max_grad_norm=0.8, # VAL 1 = 1.0, 2 and 3 and 4 = 0.8
    beta=0.1, # VAL 4 = 0.1, setting beta to 0.3 for VAL 2 and VAL 3, VAL 1 was default
    report_to="wandb",
    ddp_find_unused_parameters=False,
)

# Initialize trainer
print("Initializing DPO trainer...")

# Pass the actual tokenizer to DPOTrainer instead of the processor
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor,
    # tokenizer=tokenizer,  # Tokenizer IS NOT passed here, processor handles it
)

# Start training
print("Starting training...")
import time
start_time = time.time()

try:
    trainer.train()
    
    # Save model
    trainer.save_model("./fastvlm-dpo-final-val4")
    processor.save_pretrained("./fastvlm-dpo-final-val4")
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
    
    # Save checkpoint even if training fails
    trainer.save_model("./fastvlm-dpo-checkpoint-val4")
    processor.save_pretrained("./fastvlm-dpo-checkpoint-val4")

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
    print("Model saved to ./fastvlm-dpo-final-val4")
    
    wandb.finish()
