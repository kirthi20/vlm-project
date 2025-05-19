import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers.image_utils import load_image

# Set the device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load processor and model
model_name = "Salesforce/blip2-opt-2.7b"
print(f"Loading {model_name}...")

processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
).to(DEVICE)

# Load a test image
print("Loading test image...")
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image = load_image(image_url)

# Prepare inputs
question = "" #"Question: Describe this image briefly. Answer:"
inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE)

# Check input token count
input_token_count = inputs.input_ids.shape[1]
print(f"Input token count: {input_token_count}")
print(f"Input text decoded: {processor.decode(inputs.input_ids[0], skip_special_tokens=True)}")

# Generate output
print("Generating caption...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,  # Generate more tokens
        min_new_tokens=20,   # Ensure at least 20 new tokens
        num_beams=5,
        early_stopping=True
    )

# Check output shape
print(f"Output shape: {output_ids.shape}")

# Calculate how many new tokens were generated
new_tokens = output_ids.shape[1] - input_token_count
print(f"New tokens generated: {new_tokens}")

# Try different decoding methods
print("\n1. Full output with special tokens:")
print(processor.decode(output_ids[0], skip_special_tokens=False))

print("\n2. Full output without special tokens:")
print(processor.decode(output_ids[0], skip_special_tokens=True))

# Only decode the new tokens (excluding input)
print("\n3. Only new tokens (should be the actual generated response):")
if new_tokens > 0:
    new_token_ids = output_ids[0, input_token_count:]
    print(processor.decode(new_token_ids, skip_special_tokens=True))
else:
    print("No new tokens generated!")