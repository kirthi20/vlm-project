import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Set the device to either CPU or GPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
model_name = "Vision-CAIR/MiniGPT-4"  # Replace with the correct HuggingFace model name
print(f"Loading {model_name}...")

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32,
).to(DEVICE)

# Load a test image
print("Loading test image...")
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image = load_image(image_url)

# Create input for MiniGPT-4
# Note: MiniGPT-4 may use a different input format than SmolVLM
prompt = "Describe this image briefly."

# Prepare inputs - adjust based on MiniGPT-4's expected format
inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

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