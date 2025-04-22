"""
Dataset wrapper for HuggingFace datasets with proper COCO support
"""

import torch
import random
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import ToPILImage
import logging

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("tensor_defense")

class HFDatasetWrapper(Dataset):
    """
    Dataset wrapper for HuggingFace dataset with improved COCO support
    """
    def __init__(self, hf_dataset, split="test", max_samples=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            hf_dataset: HuggingFace dataset
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to use (None = use all)
            transform: Optional transform to apply to images
        """
        self.hf_dataset = hf_dataset[split]
        self.transform = transform
        
        # Check if this is a COCO dataset by looking at the features
        dataset_str = str(hf_dataset)
        self.is_coco = ('coco_url' in dataset_str and 'annotations' in dataset_str)
        
        # Debug dataset info
        logger.info(f"Dataset type: {type(hf_dataset)}")
        logger.info(f"Dataset string representation: {dataset_str}")
        logger.info(f"Is COCO dataset: {self.is_coco}")
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples > 0:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
        
        # For COCO, create an expanded dataset with one entry per image
        if self.is_coco:
            self.expanded_dataset = []
            for idx in range(len(self.hf_dataset)):
                item = self.hf_dataset[idx]
                if 'annotations' in item and isinstance(item['annotations'], dict):
                    annotations = item['annotations']
                    if 'caption' in annotations and isinstance(annotations['caption'], list):
                        # Use only the first caption (both COCO and Flickr have 5 captions per image)
                        caption = annotations['caption'][0]
                        self.expanded_dataset.append({
                            'original_idx': idx,
                            'caption': caption,
                            'image_id': str(item['image_id'])
                        })
                else:
                    # Fallback if no annotations found
                    self.expanded_dataset.append({
                        'original_idx': idx,
                        'caption': "",
                        'image_id': str(item.get('image_id', idx))
                    })
            
            logger.info(f"Created {len(self.expanded_dataset)} image-caption pairs from {len(self.hf_dataset)} images")
        
        logger.info(f"Initialized dataset with {len(self)} samples")
    
    def __len__(self):
        """Return the total number of samples"""
        if self.is_coco and hasattr(self, 'expanded_dataset'):
            return len(self.expanded_dataset)
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """Get an item from the dataset"""
        if self.is_coco and hasattr(self, 'expanded_dataset'):
            # Get the expanded item (which contains original_idx and caption info)
            expanded_item = self.expanded_dataset[idx]
            original_idx = expanded_item['original_idx']
            caption = expanded_item['caption']
            image_id = expanded_item['image_id']
            
            # Get the original item to access the image
            item = self.hf_dataset[original_idx]
        else:
            # For non-COCO datasets (like Flickr), use the index directly
            item = self.hf_dataset[idx]
            
            # Get caption - both COCO and Flickr have multiple captions per image
            if 'caption' in item:
                caption_data = item['caption']
                # Use only the first caption (both datasets have 5 captions per image)
                caption = caption_data[0] if isinstance(caption_data, list) else caption_data
            elif 'text' in item:
                caption = item['text']
            else:
                caption = ""  # Empty caption as fallback
                
            # Get image ID
            if 'image_id' in item:
                image_id = str(item['image_id'])
            elif 'filename' in item:
                image_id = item['filename']
            elif 'img_id' in item:
                image_id = str(item['img_id'])
            else:
                image_id = str(idx)
        
        # Get image - should be a PIL Image or convertible to one
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        elif 'pixel_values' in item:
            # Convert tensor to PIL image if needed
            if isinstance(item['pixel_values'], torch.Tensor):
                image = ToPILImage()(item['pixel_values'].squeeze())
            else:
                image = item['pixel_values']
        else:
            logger.error(f"No image field found. Available keys: {item.keys()}")
            # Create a blank image as fallback
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Ensure we have a PIL image
        if not isinstance(image, Image.Image):
            try:
                if isinstance(image, torch.Tensor):
                    image = ToPILImage()(image)
                else:
                    image = Image.fromarray(image)
            except Exception as e:
                logger.error(f"Error converting image to PIL: {str(e)}")
                # Return a blank image as fallback to avoid breaking the dataset
                image = Image.new('RGB', (224, 224), color='gray')
        
        # Apply transform if specified
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image_id': image_id,
            'image': image,
            'caption': caption
        }


# Custom collate function to handle PIL images
def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images
    
    Args:
        batch: Batch of samples
        
    Returns:
        Dictionary of batched data
    """
    image_ids = [item['image_id'] for item in batch]
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    return {
        'image_id': image_ids,
        'image': images,
        'caption': captions
    }