import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Test different CUDA devices
print(range(torch.cuda.device_count()))
for device_id in [1, 2, 3]:
    device = f"cuda:{device_id}"
    print(f"\n--- Testing {device} ---")
    
    try:
        torch.cuda.set_device(device_id)  # Set active device
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
        print(f"✓ Success on {device}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Failed on {device}: {e}")