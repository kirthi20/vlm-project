import torch
from PIL import Image
#from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Set the device to either CPU or GPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
print(f"Loading {model_name}...")

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
).to(DEVICE)

# Load a test image
print("Loading test image...")
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image = load_image(image_url)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image briefly."}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate output
print("Generating caption...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

# Decode and print the result
result = processor.decode(output[0], skip_special_tokens=True)
print("\nModel output:")
print(result)