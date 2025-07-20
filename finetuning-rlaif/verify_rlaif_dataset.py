from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from datasets import load_dataset


model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# Load dataset
dataset = load_dataset("HuggingFaceH4/rlaif-v_formatted")

# Preprocessing function
def preprocess_function(examples):
    # Extract chosen and rejected responses
    chosen_images = examples["chosen"]
    rejected_images = examples["rejected"]
    
    # Process chosen examples
    chosen_inputs = processor(
        text=examples["chosen"],
        images=chosen_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Process rejected examples
    rejected_inputs = processor(
        text=examples["rejected"],
        images=rejected_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    return {
        "input_ids_chosen": chosen_inputs["input_ids"],
        "attention_mask_chosen": chosen_inputs["attention_mask"],
        "pixel_values_chosen": chosen_inputs["pixel_values"],
        "input_ids_rejected": rejected_inputs["input_ids"],
        "attention_mask_rejected": rejected_inputs["attention_mask"],
        "pixel_values_rejected": rejected_inputs["pixel_values"],
    }

# Preprocess dataset
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(processed_dataset["train"][0])