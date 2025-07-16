import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import gc
import time
from datetime import datetime
import wandb
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig
from transformers.image_utils import load_image

# Set up environment
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize wandb
wandb.init(project="fastvlm-qlora-dpo-finetuning", mode="online")

# GPU setup
torch.cuda.set_device(0)  # GPU 3 is now referred to as cuda:0
DEVICE = torch.device("cuda:0")
device_map = {"": 0}  # or device_map={"": torch.cuda.current_device()}

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")
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

class FastVLMProcessor:
    """Enhanced processor for DPO training compatibility"""
    
    def __init__(self, tokenizer, image_processor, model_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        
        # Add required attributes for DPO trainer compatibility
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        # Set image processing parameters
        self.image_processor.size = {"longest_edge": 512}
        self.image_processor.max_image_size = {"longest_edge": 512}
        
    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        """Apply chat template for conversation formatting"""
        conv = conv_templates["plain"].copy()

        # Debug the conversation template thoroughly
        print(f"Conv template debug:")
        print(f"  roles: {conv.roles}")
        print(f"  seps: {conv.seps}")
        print(f"  messages: {conv.messages}")
        
        # Check if any None values in seps
        if conv.seps:
            for i, sep in enumerate(conv.seps):
                if sep is None:
                    print(f"  WARNING: sep[{i}] is None!")
                    conv.seps[i] = ""  # Replace None with empty string
        
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            print(f"Message {i}: role={role}, content={content}")  # Debug print
            
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
                print(f"User text after processing: {text}")  # Debug print
                
                if has_image:
                    if self.model_config.mm_use_im_start_end:
                        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
                    else:
                        text = DEFAULT_IMAGE_TOKEN + '\n' + text
                
                if text is not None:
                    conv.append_message(conv.roles[0], text)
                else:
                    print("WARNING: User text is None, using empty string")
                    conv.append_message(conv.roles[0], "")
                    
            elif role == "assistant":
                # Handle assistant content that might be a list
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = " ".join(text_parts)
                
                print(f"Assistant content after processing: {content}")  # Debug print
                
                if content is not None:
                    conv.append_message(conv.roles[1], content)
                else:
                    print("WARNING: Assistant content is None, using empty string")
                    conv.append_message(conv.roles[1], "")
        
        # Get the prompt without the generation prompt first
        prompt = conv.get_prompt()
        
        # Manually add the generation prompt if needed
        if add_generation_prompt:
            prompt += f"{conv.roles[1]}: "
        
        return prompt
    
    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        """Process inputs for training"""
        # Handle different input formats
        if isinstance(text, list):
            # Batch processing
            batch_input_ids = []
            batch_attention_masks = []
            batch_images = []
            
            for i, single_text in enumerate(text):
                single_image = images[i] if images and i < len(images) else None
                
                # Process single item
                result = self._process_single(single_text, single_image, return_tensors)
                batch_input_ids.append(result["input_ids"])
                batch_attention_masks.append(result["attention_mask"])
                if result["images"] is not None:
                    batch_images.append(result["images"])
            
            # Stack tensors
            max_length = max(ids.shape[-1] for ids in batch_input_ids)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(batch_input_ids, batch_attention_masks):
                pad_length = max_length - ids.shape[-1]
                if pad_length > 0:
                    padded_ids = torch.cat([ids, torch.full((ids.shape[0], pad_length), self.pad_token_id)], dim=1)
                    padded_mask = torch.cat([mask, torch.zeros(mask.shape[0], pad_length)], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            return {
                "input_ids": torch.cat(padded_input_ids, dim=0),
                "attention_mask": torch.cat(padded_attention_masks, dim=0),
                "images": torch.cat(batch_images, dim=0) if batch_images else None
            }
        else:
            # Single item processing
            return self._process_single(text, images, return_tensors)
    
    def _process_single(self, text, images, return_tensors):
        """Process a single text-image pair"""
        # Process images
        if images:
            if not isinstance(images, list):
                images = [images]
            
            # Ensure images are PIL Images and RGB
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    img = load_image(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                processed_images.append(img)
            
            image_tensor = process_images(processed_images, self.image_processor, self.model_config)
        else:
            image_tensor = None
        
        # Process text
        input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=return_tensors)
        
        # Ensure proper dimensions
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image_tensor
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode tokens"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def save_pretrained(self, path):
        """Save processor components"""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        # Note: image_processor and model_config would need custom serialization


class FastVLMModelWrapper(nn.Module):
    """Enhanced model wrapper for DPO training - inheriting from nn.Module"""
    
    def __init__(self, model, tokenizer):
        super().__init__()  # Initialize nn.Module
        self.model = model  # This automatically registers as a submodule
        self.tokenizer = tokenizer
        self.config = model.config
        
        # Explicitly set common attributes that PEFT might look for
        self.model_type = getattr(model, 'model_type', None)
        self.base_model_prefix = getattr(model, 'base_model_prefix', '')
        self.name_or_path = getattr(model, 'name_or_path', '')
        
        # Note: device and dtype are now handled automatically by nn.Module
        self.warnings_issued = {}
        self.generation_config = getattr(model, 'generation_config', None)
        
    def forward(self, input_ids=None, attention_mask=None, images=None, labels=None, **kwargs):
        """Forward pass compatible with training frameworks"""
        # Get device from the model (nn.Module handles this automatically)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Prepare inputs
        if images is not None:
            if isinstance(images, list):
                images = [img.to(device, dtype=dtype) for img in images]
            else:
                images = images.to(device, dtype=dtype)
        
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Call the original model's forward method
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids=None, images=None, max_new_tokens=512, do_sample=False, **kwargs):
        """Generate method for inference"""
        # Get device from the model
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Prepare inputs
        if images is not None:
            if isinstance(images, list):
                images = [img.to(device, dtype=dtype) for img in images]
            else:
                images = images.to(device, dtype=dtype)
        
        input_ids = input_ids.to(device)
        
        # Generate using FastVLM's approach
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **kwargs
            )
        
        return output_ids
    
    # These methods are now much simpler since nn.Module handles the heavy lifting
    def half(self):
        """Convert to half precision"""
        super().half()
        return self
    
    def float(self):
        """Convert to float precision"""
        super().float()
        return self
    
    def cuda(self, device=None):
        """Move to CUDA"""
        super().cuda(device)
        return self
    
    def cpu(self):
        """Move to CPU"""
        super().cpu()
        return self
    
    # Remove these methods - nn.Module provides them automatically:
    # - modules()
    # - named_modules()
    # - children()
    # - named_children()
    # - parameters()
    # - named_parameters()
    # - state_dict()
    # - load_state_dict()
    # - to()
    # - train()
    # - eval()
    # - zero_grad()
    # - apply()

    def set_input_embeddings(self, value):
        """Set input embeddings on the underlying model"""
        self.model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        """Set output embeddings on the underlying model"""
        self.model.set_output_embeddings(value)

    def get_base_model(self):
        """Get the base model - useful for PEFT"""
        return self.model
    
    def get_input_embeddings(self):
        """Get input embeddings from the underlying model"""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        """Get output embeddings from the underlying model"""
        return self.model.get_output_embeddings()
    
    # Optional: Add a property to access the device easily
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def main_input_name(self):
        """Main input name for the model"""
        return getattr(self.model, 'main_input_name', 'input_ids')

    @property  
    def model_parallel(self):
        """Model parallel flag"""
        return getattr(self.model, 'model_parallel', False)

    @property
    def is_parallelizable(self):
        """Whether model is parallelizable"""
        return getattr(self.model, 'is_parallelizable', False)

    @property
    def supports_gradient_checkpointing(self):
        """Whether model supports gradient checkpointing"""
        return getattr(self.model, 'supports_gradient_checkpointing', True)
    
def load_fastvlm_model(model_path, device):
    """Load FastVLM model with training compatibility"""
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    
    # Load pretrained model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    
    # Create processor and wrapped model
    processor = FastVLMProcessor(tokenizer, image_processor, model.config)
    wrapped_model = FastVLMModelWrapper(model, tokenizer)
    
    return processor, wrapped_model


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


# Load FastVLM model
model_path = "checkpoints/llava-fastvithd_0.5b_stage3"
print(f"Loading FastVLM model from {model_path}...")

try:
    processor, model = load_fastvlm_model(model_path, DEVICE)
    model = model.to(DEVICE)
    if DEVICE.type == 'cuda':
        model = model.half()  # nn.Module handles this properly now
    
    print("FastVLM model loaded successfully!")
    
except Exception as e:
    print(f"Error loading FastVLM model: {e}")
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

# Prepare model for k-bit training if using quantization
# model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load and preprocess dataset
print("Loading dataset...")
train_dataset = load_dataset(
    "HuggingFaceH4/rlaif-v_formatted",
    split="train"
).take(100)

# Apply preprocessing
train_dataset = train_dataset.map(ensure_rgb_and_resize, num_proc=16)

# Training configuration
training_args = DPOConfig(
    output_dir="./fastvlm-dpo-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduced for FastVLM
    gradient_accumulation_steps=16,  # Effective batch size = 16
    gradient_checkpointing=True,
    optim="adamw_torch",
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=True if DEVICE.type == 'cuda' else False,
    fp16=False,
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
start_time = time.time()

try:
    trainer.train()
    
    # Save model
    trainer.save_model("./fastvlm-dpo-final")
    processor.save_pretrained("./fastvlm-dpo-final")
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    # Save checkpoint even if training fails
    trainer.save_model("./fastvlm-dpo-checkpoint")
    processor.save_pretrained("./fastvlm-dpo-checkpoint")

finally:
    # Clean up
    if DEVICE.type == 'cuda':
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
