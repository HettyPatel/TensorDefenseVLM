"""
Model loading utilities for vision-language models
"""

import torch
import logging
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForImageTextRetrieval
)

logger = logging.getLogger("tensor_defense")

def load_model(model_name, variant, device="cuda"):
    """
    Load a vision-language model based on configuration
    
    Args:
        model_name: Name of the model architecture (clip, blip)
        variant: Specific variant of the model
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        processor: Corresponding processor
    """
    try:
        # Determine which model to load
        if model_name.lower() == "clip":
            model_id = f"openai/clip-{variant.lower()}"
            model = CLIPModel.from_pretrained(model_id).to(device)
            processor = CLIPProcessor.from_pretrained(model_id)
            logger.info(f"Loaded CLIP model: {model_id}")
        
        elif model_name.lower() == "blip":
            if "base" in variant.lower():
                model_id = "Salesforce/blip-image-captioning-base"
            elif "large" in variant.lower():
                model_id = "Salesforce/blip-image-captioning-large"
            else:
                model_id = f"Salesforce/blip-{variant.lower()}"
                
            model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device)
            processor = BlipProcessor.from_pretrained(model_id)
            logger.info(f"Loaded BLIP model: {model_id}")
        
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
            
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model {model_name} {variant}: {str(e)}")
        raise e