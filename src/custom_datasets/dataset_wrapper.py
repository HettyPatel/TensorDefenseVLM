"""
Dataset wrapper for HuggingFace datasets
"""

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import ToPILImage
import logging

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("tensor_defense")

class HFDatasetWrapper(Dataset):
    """
    Dataset wrapper for HuggingFace dataset
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
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples > 0:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
            
        logger.info(f"Initialized dataset with {len(self.hf_dataset)} samples")
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
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
            # Handle dataset-specific image fields
            if 'flickr30k' in str(self.hf_dataset):
                image = item['image']  # For Flickr30k
            elif 'coco' in str(self.hf_dataset):
                image = item['image']  # For COCO
            else:
                raise ValueError(f"No image field found in dataset item: {item.keys()}")
        
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
        
        # Get caption (use first caption if there are multiple)
        if 'caption' in item:
            captions = item['caption']
            caption = captions[0] if isinstance(captions, list) else captions
        elif 'text' in item:
            caption = item['text']
        elif 'captions' in item:
            # For dataset-specific handling
            captions = item['captions']
            caption = captions[0] if isinstance(captions, list) else captions
        else:
            caption = ""  # Empty caption as fallback
        
        # Get image id 
        if 'filename' in item:
            image_id = item['filename']
        elif 'img_id' in item:
            image_id = str(item['img_id'])
        elif 'image_id' in item:
            image_id = str(item['image_id'])
        else:
            image_id = str(idx)
        
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