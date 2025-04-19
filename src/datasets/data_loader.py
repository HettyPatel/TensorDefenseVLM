"""
Dataset loaders for COCO and Flickr30k datasets.

This module provides utilities for loading and processing image-text datasets
for use in adversarial attack experiments on VLMs.
"""

import torch
import logging
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import ToPILImage

logger = logging.getLogger("tensor_defense")

class VLMDatasetWrapper(Dataset):
    """
    Dataset wrapper for Vision-Language Model datasets
    
    This class provides a consistent interface for HuggingFace datasets
    to be used in adversarial attack experiments.
    """
    def __init__(self, dataset_name, split="test", max_samples=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            dataset_name: HuggingFace dataset name ('nlphuji/flickr30k', 'laion/coco')
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to use (None = use all)
            transform: Optional transform to apply to images
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        # Load the dataset
        try:
            self.hf_dataset = load_dataset(dataset_name)
            
            # Get the appropriate split
            if split in self.hf_dataset:
                self.data = self.hf_dataset[split]
            else:
                # Some datasets have different split names
                available_splits = list(self.hf_dataset.keys())
                logger.warning(f"Split '{split}' not found. Available splits: {available_splits}")
                
                # Try to find a reasonable default
                if "validation" in available_splits:
                    self.data = self.hf_dataset["validation"]
                    logger.info(f"Using 'validation' split instead of '{split}'")
                elif "test" in available_splits:
                    self.data = self.hf_dataset["test"]
                    logger.info(f"Using 'test' split instead of '{split}'")
                elif len(available_splits) > 0:
                    self.data = self.hf_dataset[available_splits[0]]
                    logger.info(f"Using '{available_splits[0]}' split instead of '{split}'")
                else:
                    raise ValueError(f"No suitable split found in dataset {dataset_name}")
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples > 0:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
            
        logger.info(f"Initialized dataset {dataset_name} with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image - handle different dataset structures
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
            # Try to find any image-like field
            image_keys = [k for k in item.keys() if 'image' in k.lower() or 'img' in k.lower()]
            if image_keys:
                image = item[image_keys[0]]
            else:
                raise ValueError(f"No image field found in dataset item: {item.keys()}")
        
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
            captions = item['captions']
            caption = captions[0] if isinstance(captions, list) else captions
        else:
            # Try to find any text-like field
            text_keys = [k for k in item.keys() if 'text' in k.lower() or 'caption' in k.lower()]
            if text_keys:
                caption = item[text_keys[0]]
                if isinstance(caption, list):
                    caption = caption[0]
            else:
                caption = ""  # Empty caption as fallback
        
        # Get image ID for tracking
        if 'image_id' in item:
            image_id = str(item['image_id'])
        elif 'img_id' in item:
            image_id = str(item['img_id'])
        elif 'id' in item:
            image_id = str(item['id'])
        elif 'filename' in item:
            image_id = item['filename']
        else:
            image_id = str(idx)
        
        return {
            'image_id': image_id,
            'image': image,
            'caption': caption
        }


# Custom collate function to handle PIL images and varying data types
def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images and text
    
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


def get_dataloader(dataset_name, split="test", batch_size=32, max_samples=None, 
                  num_workers=4, transform=None, shuffle=False):
    """
    Create a DataLoader for the specified dataset
    
    Args:
        dataset_name: HuggingFace dataset name ('nlphuji/flickr30k', 'laion/coco')
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size for the dataloader
        max_samples: Maximum number of samples to use
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to images
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = VLMDatasetWrapper(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    return dataloader