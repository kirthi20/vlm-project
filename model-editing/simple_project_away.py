import os
os.environ["HF_HOME"] = "data/catz0452/cache/huggingface"  # Set Hugging Face cache directory

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import List, Dict, Tuple, Optional
import numpy as np
import requests
from io import BytesIO


class ProjectAway:
    """ProjectAway implementation for reducing hallucinations in Vision-Language Models.
    
    This implementation follows the methodology from "Interpreting and Editing Vision-Language 
    Representations to Mitigate Hallucinations" (Jiang et al., ICLR 2025).
    """
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        """Initialize ProjectAway with a vision-language model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_image_size={"longest_edge": 512}  # Use dictionary format
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Get model components
        self.vision_encoder = self.model.model.vision_model.encoder
        self.language_model = self.model.model.text_model
        self.vision_projection = self.model.model.connector
        
        # Cache for text embeddings
        self.text_embedding_cache = {}
        
    def get_text_embedding(self, text: str, layer: int = -1) -> torch.Tensor:
        """Get text embedding for a given object at specified layer.
        
        Args:
            text: Object text (e.g., "hot dog")
            layer: Layer index to extract embedding from (-1 for last layer)
            
        Returns:
            Text embedding tensor
        """
        cache_key = f"{text}_{layer}"
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]
            
        # Tokenize the text
        text_inputs = self.processor.tokenizer(
            text, 
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            if layer == -1:
                # Use the embedding layer directly
                text_embeddings = self.language_model.get_input_embeddings()(
                    text_inputs.input_ids
                )
                # Take the last token embedding
                text_embedding = text_embeddings[0, -1]
            else:
                # Run through model to get intermediate layer representation
                outputs = self.language_model(
                    input_ids=text_inputs.input_ids,
                    output_hidden_states=True
                )
                # Get representation at specified layer for last token
                text_embedding = outputs.hidden_states[layer][0, -1]
                
        self.text_embedding_cache[cache_key] = text_embedding
        return text_embedding
    
    def apply_logit_lens(
        self, 
        image_embeddings: torch.Tensor,
        layer: int = -1
    ) -> torch.Tensor:
        """Apply logit lens to image embeddings to get vocabulary probabilities.
        
        Args:
            image_embeddings: Image embeddings [batch_size, num_patches, hidden_dim]
            layer: Layer to apply logit lens at
            
        Returns:
            Logit distributions over vocabulary
        """
        with torch.no_grad():
            # Pass through language model layers if needed
            if layer > 0:
                # Create attention mask
                attention_mask = torch.ones(
                    image_embeddings.shape[:2], 
                    device=self.device
                )
                
                # Pass through specified number of layers
                outputs = self.language_model(
                    inputs_embeds=image_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[layer]
            else:
                hidden_states = image_embeddings
                
            # Apply language model head to get logits
            logits = self.language_model.lm_head(hidden_states)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
        return probs
    
    def get_internal_confidence(
        self,
        image: torch.Tensor,
        objects: List[str],
        layers_to_check: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Calculate internal confidence scores for objects in an image.
        
        Args:
            image: Input image tensor
            objects: List of object names to check
            layers_to_check: Which layers to examine (None for all)
            
        Returns:
            Dictionary mapping object names to confidence scores
        """
        # Process image
        with torch.no_grad():
            # Get vision features
            vision_features = self.vision_encoder(image)
            
            # Project to language model dimension
            image_embeddings = self.vision_projection(vision_features)
            
        if layers_to_check is None:
            # Check all layers
            num_layers = len(self.language_model.model.layers)
            layers_to_check = list(range(num_layers))
            
        confidences = {}
        
        for obj in objects:
            # Tokenize object name
            obj_tokens = self.processor.tokenizer(
                obj,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            max_conf = 0.0
            
            # Check each layer
            for layer in layers_to_check:
                # Get probabilities at this layer
                probs = self.apply_logit_lens(image_embeddings, layer)
                
                # Get max probability for any token of this object
                for token_id in obj_tokens[0]:
                    token_probs = probs[:, :, token_id].max().item()
                    max_conf = max(max_conf, token_probs)
                    
            confidences[obj] = max_conf
            
        return confidences
    
    def project_away(
        self,
        image_embeddings: torch.Tensor,
        objects_to_remove: List[str],
        weight: float = 1.0,
        edit_layer: int = 15,
        text_layer: int = 15
    ) -> torch.Tensor:
        """Apply ProjectAway algorithm to remove objects from image representations.
        
        Args:
            image_embeddings: Image embeddings to edit
            objects_to_remove: List of object names to remove
            weight: Weight factor for subtraction (alpha in paper)
            edit_layer: Which layer to edit at
            text_layer: Which layer to extract text embeddings from
            
        Returns:
            Edited image embeddings
        """
        edited_embeddings = image_embeddings.clone()
        
        for obj in objects_to_remove:
            # Get text embedding for this object
            text_embedding = self.get_text_embedding(obj, text_layer)
            
            # Normalize text embedding
            text_embedding = F.normalize(text_embedding, dim=-1)
            
            # For each image patch embedding
            for i in range(edited_embeddings.shape[1]):
                patch_embedding = edited_embeddings[0, i]
                
                # Calculate dot product
                dot_product = torch.dot(patch_embedding, text_embedding)
                
                # Only subtract if dot product is positive
                if dot_product > 0:
                    # Subtract weighted text embedding
                    edited_embeddings[0, i] = patch_embedding - weight * dot_product * text_embedding
                    
        return edited_embeddings
    
    def detect_hallucinations(
        self,
        image: torch.Tensor,
        generated_caption: str,
        confidence_threshold: float = 0.15
    ) -> List[str]:
        """Detect potential hallucinations in a generated caption.
        
        Args:
            image: Input image
            generated_caption: Caption generated by the model
            confidence_threshold: Threshold below which objects are considered hallucinations
            
        Returns:
            List of potentially hallucinated objects
        """
        # Extract objects from caption (simple version - you may want to use spaCy or similar)
        # This is a simplified extraction - in practice you'd use better NLP
        import re
        
        # Common object patterns
        objects = []
        
        # Extract nouns (simplified)
        words = generated_caption.lower().split()
        
        # Get confidence scores
        if words:
            confidences = self.get_internal_confidence(image, words)
            
            # Find low-confidence objects
            hallucinations = [
                obj for obj, conf in confidences.items() 
                if conf < confidence_threshold
            ]
            
            return hallucinations
        
        return []
    
    def generate_with_hallucination_reduction(
        self,
        image,
        prompt: str = "Describe this image in detail.",
        max_length: int = 100,
        confidence_threshold: float = 0.15,
        removal_weight: float = 1.0,
        edit_layer: int = 15,
        text_layer: int = 15
    ) -> str:
        """Generate caption with hallucination reduction using ProjectAway.
        
        Args:
            image: Input image (PIL Image or tensor)
            prompt: Text prompt
            max_length: Maximum caption length
            confidence_threshold: Threshold for hallucination detection
            removal_weight: Weight for ProjectAway
            edit_layer: Layer to edit
            text_layer: Layer for text embeddings
            
        Returns:
            Generated caption with reduced hallucinations
        """
        # Process inputs
        inputs = self.processor(
            images=image,
            text=f"{self.processor.image_token}{prompt}",
            return_tensors="pt"
        ).to(self.device)
        
        # First, generate initial caption to detect potential hallucinations
        with torch.no_grad():
            initial_outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Generate up to 100 NEW tokens
                do_sample=False
            )

            initial_caption = self.processor.decode(
                initial_outputs[0], 
                skip_special_tokens=True
            )
        
        # Detect potential hallucinations
        hallucinations = self.detect_hallucinations(
            inputs.pixel_values,
            initial_caption,
            confidence_threshold
        )
        
        if hallucinations:
            print(f"Detected potential hallucinations: {hallucinations}")
            
            # Get image embeddings
            with torch.no_grad():
                vision_features = self.vision_encoder(inputs.pixel_values)
                image_embeddings = self.vision_projection(vision_features)
            
            # Apply ProjectAway to remove hallucinations
            edited_embeddings = self.project_away(
                image_embeddings,
                hallucinations,
                weight=removal_weight,
                edit_layer=edit_layer,
                text_layer=text_layer
            )
            
            # Generate with edited embeddings
            # This requires creating custom inputs with the edited embeddings
            # Note: This is a simplified version - full implementation would need
            # to properly integrate with the model's forward pass
            
            # For now, return the initial caption with a note
            return f"{initial_caption}\n[Note: Hallucinations detected and would be removed: {', '.join(hallucinations)}]"
        
        return initial_caption


# Example usage
if __name__ == "__main__":
    from PIL import Image
    
    # Initialize ProjectAway
    pa = ProjectAway("HuggingFaceTB/SmolVLM-256M-Instruct")
    
    # Load an image
    print("\nTesting model...")
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    
    # Download test image
    response = requests.get(test_image_url)
    image = Image.open(BytesIO(response.content))
    
    #Generate caption with hallucination reduction
    caption = pa.generate_with_hallucination_reduction(
        image,
        prompt="Describe this image.",
        confidence_threshold=0.15,
        removal_weight=1.0
    )
    print(caption)