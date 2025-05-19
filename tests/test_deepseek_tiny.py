import torch
# import sys
# import os
from PIL import Image
# import requests
# from io import BytesIO

# Add the DeepSeek-VL repo to the Python path
# Adjust this path to where you've cloned the repository
# deepseek_vl_path = "/path/to/DeepSeek-VL"  # Replace with your actual path
# sys.path.append(deepseek_vl_path)

from transformers import AutoModelForCausalLM
from transformers.image_utils import load_image

# Import directly from the downloaded repo
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images

# Set the device
DEVICE =  "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load model and processor
# model_name = "deepseek-ai/deepseek-vl2-tiny"
# print(f"Loading {model_name}...")

# # Initialize processor and model directly using the local modules
# processor = DeepseekVLV2Processor.from_pretrained(model_name)
# model = DeepseekVLV2ForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32,
# ).to(DEVICE)
# model.eval()

model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Load a test image
print("Loading test image...")
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image = load_image(image_url)

# Create a conversation in the format expected by DeepSeek VL2
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\nDescribe this image briefly.",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# Process the conversation
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

