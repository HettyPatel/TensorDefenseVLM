"""
Script to inspect COCO dataset structure
"""

import os
import json
from datasets import load_dataset
import logging
from PIL import Image
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coco_inspector")

def inspect_coco_dataset():
    """Inspect the COCO dataset structure"""
    logger.info("Loading COCO dataset...")
    
    # Load the dataset
    dataset = load_dataset(
        "shunk031/MSCOCO",
        year=2014,
        coco_task="captions",
        trust_remote_code=True
    )
    
    # Print dataset splits
    logger.info(f"Dataset splits: {dataset.keys()}")
    
    # Create directory for samples
    os.makedirs("coco_samples_train", exist_ok=True)
    os.makedirs("coco_samples_validation", exist_ok=True)
    
    # Inspect each split
    for split in ['train', 'validation']:
        logger.info(f"\n === Inspecting '{split}' split ===")
        split_data = dataset[split]
        logger.info(f"Total items in {split}: {len(split_data)}")
        
        # Sample some items to inspect
        sample_indices = random.sample(range(len(split_data)), 5)
        
        for idx in sample_indices:
            item = split_data[idx]
            logger.info(f"\nItem #{idx}:")
            logger.info(f"Keys: {sorted(item.keys())}")
            
            # Save a sample image
            if 'image' in item:
                image = item['image']
                if isinstance(image, Image.Image):
                    image.save(f"coco_samples_{split}/sample_{idx}.jpg")
            
            # Save the structure
            with open(f"coco_samples_{split}/structure_{idx}.json", 'w') as f:
                # Convert PIL image to string for JSON serialization
                item_copy = item.copy()
                if 'image' in item_copy:
                    item_copy['image'] = f"PIL Image: {item_copy['image'].size}"
                json.dump(item_copy, f, indent=2)

if __name__ == "__main__":
    inspect_coco_dataset() 