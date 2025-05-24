from datasets import load_dataset

print("Loading dataset...")
# This will automatically download and cache everything for you
dataset = load_dataset("yerevann/coco-karpathy")

# Print the available splits
print("Available splits:", dataset.keys())
# Access the splits
val_data = dataset['validation'] 

# Print the number of examples in the validation set
print("Number of examples in validation set:", len(val_data))
# Access the first example in the validation set
first_example = val_data[10]

print("First example structure:")
print(first_example)
print("\n" + "="*50 + "\n")

# Access the image URL and captions (note: it's 'sentences', not 'caption')
image_url = first_example['url']
captions = first_example['sentences']  # This is a list of 5 captions
filename = first_example['filename']
split_type = first_example['split']

# Print the details
print(f"Split: {split_type}")
print(f"Filename: {filename}")
print(f"Image URL: {image_url}")
print(f"Number of captions: {len(captions)}")
print("Captions:")
for i, caption in enumerate(captions, 1):
    print(f"  {i}. {caption}")

# If you want to actually load and display the image:
from PIL import Image
import requests
from io import BytesIO

# Download and display the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
print(f"\nImage size: {image.size}")

image.save("tests/sample_coco_image.jpg")
print("Image saved as 'sample_coco_image.jpg'.")