"""
Evaluation utilities for measuring defense effectiveness.

This module provides functions for evaluating how well tensor decomposition
defenses protect VLMs against adversarial attacks.
"""

import torch
import numpy as np
import logging
import pandas as pd
import os
from tqdm import tqdm

logger = logging.getLogger("tensor_defense")

def calculate_recall_at_k(similarity, targets, k=1):
    """
    Calculate recall@k from similarity matrix
    
    Args:
        similarity: Similarity matrix of shape [batch_size, batch_size]
        targets: Ground truth indices
        k: k value for recall@k
    
    Returns:
        Recall@k value
    """
    # Ensure k is at least 1 and at most the width of the similarity matrix
    k = max(1, min(k, similarity.shape[1]))
    
    # Get top-k indices
    _, indices = similarity.topk(k, dim=1)
    correct = torch.any(indices == targets.view(-1, 1), dim=1).float()
    recall = correct.mean().item()
    return recall


def generate_adversarial_examples(model, processor, dataloader, attack, device, max_batches=None):
    """
    Generate adversarial examples for dataset evaluation
    
    Args:
        model: Model to attack
        processor: Image/text processor
        dataloader: DataLoader for evaluation data
        attack: Attack instance (PGD or FGSM)
        device: Device to run attack on
        max_batches: Maximum number of batches to process (None = all)
        
    Returns:
        Dictionary of adversarial examples and corresponding inputs
    """
    logger.info(f"Generating adversarial examples")
    
    # Storage for adversarial examples
    adv_examples = {}
    
    # Track the first batch for sample images
    sample_originals = None
    sample_adversarials = None
    sample_captions = None
    
    # Batch processing with progress bar
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating adversarial examples")):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        try:
            image_ids = batch['image_id']
            images = batch['image']
            captions = batch['caption']
            
            # Generate adversarial examples
            perturbed_pixel_values, inputs, original_pixels = attack.perturb(images, captions, device)
            
            # Store the first batch for sample images
            if batch_idx == 0:
                sample_originals = original_pixels
                sample_adversarials = perturbed_pixel_values
                sample_captions = captions
            
            # Process clean images to get text embeddings (needed later)
            with torch.no_grad():
                # For CLIP-like models
                if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
                    clean_image_embeds = model.get_image_features(pixel_values=original_pixels)
                    text_embeds = model.get_text_features(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
                    
                    # Normalize embeddings
                    clean_image_embeds = clean_image_embeds / clean_image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate clean similarity scores
                    clean_similarity = torch.matmul(clean_image_embeds, text_embeds.t())
                
                # For other model types
                else:
                    # Create inputs dictionary with the right keys
                    input_dict = {}
                    if hasattr(inputs, 'input_ids'):
                        input_dict['input_ids'] = inputs.input_ids
                    if hasattr(inputs, 'attention_mask'):
                        input_dict['attention_mask'] = inputs.attention_mask
                    if hasattr(inputs, 'pixel_values'):
                        input_dict['pixel_values'] = original_pixels
                    elif 'pixel_values' in inputs:
                        input_dict['pixel_values'] = original_pixels
                    else:
                        input_dict['images'] = original_pixels
                    
                    # Forward pass with clean images
                    clean_outputs = model(**input_dict)
                    
                    # Extract similarity or use appropriate output
                    if hasattr(clean_outputs, 'logits_per_image'):
                        clean_similarity = clean_outputs.logits_per_image
                    elif hasattr(clean_outputs, 'similarity_scores'):
                        clean_similarity = clean_outputs.similarity_scores
                    else:
                        # Fallback for other output types
                        logger.warning("Could not find explicit similarity scores, using embeddings if available")
                        if hasattr(clean_outputs, 'image_embeds') and hasattr(clean_outputs, 'text_embeds'):
                            image_embeds = clean_outputs.image_embeds
                            text_embeds = clean_outputs.text_embeds
                            
                            # Normalize embeddings
                            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                            
                            # Calculate similarity
                            clean_similarity = torch.matmul(image_embeds, text_embeds.t())
                        else:
                            logger.error("Could not calculate similarity scores from model outputs")
                            clean_similarity = torch.zeros((len(images), len(images)), device=device)
                    
                    # Store text embeddings if available
                    if hasattr(clean_outputs, 'text_embeds'):
                        text_embeds = clean_outputs.text_embeds
                    else:
                        text_embeds = None
                
                # Store in dictionary
                adv_examples[batch_idx] = {
                    'image_ids': image_ids,
                    'perturbed_pixel_values': perturbed_pixel_values.cpu(),  # Move to CPU to save GPU memory
                    'inputs': {
                        key: value.cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in (inputs.items() if hasattr(inputs, 'items') else vars(inputs).items())
                    },
                    'text_embeds': text_embeds.cpu() if text_embeds is not None else None,
                    'clean_similarity': clean_similarity.cpu()
                }
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
            
        # Free up memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logging.info(f"Generated adversarial examples for {len(adv_examples)} batches")
    
    return adv_examples, (sample_originals, sample_adversarials, sample_captions)


def evaluate_defense(model, processor, dataloader, adv_examples, defense=None, 
                    defense_name="No Defense", device=None, max_batches=None):
    """
    Evaluate model with tensor decomposition defense
    
    Args:
        model: Model to evaluate
        processor: Image/text processor
        dataloader: DataLoader for evaluation data
        adv_examples: Pre-generated adversarial examples
        defense: Defense configuration or None
        defense_name: Name of the defense for printing
        device: Device to run evaluation on
        max_batches: Maximum number of batches to process (None = all)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For storing results
    all_recalls_clean = {1: [], 5: [], 10: []}
    all_recalls_adv = {1: [], 5: [], 10: []}
    all_recalls_defended = {1: [], 5: [], 10: []}
    
    # For tracking metrics per batch
    batch_metrics = []
    
    batch_count = 0
    
    # Metrics to track
    metrics = {
        'image_ids': [],
        'defense_name': defense_name,
    }
    
    # Empty CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Apply defense if specified
    defense_instance = None
    if defense is not None:
        defense_instance = defense
    
    # Batch processing with progress bar
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {defense_name}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        # Skip if this batch wasn't in adv_examples
        if batch_idx not in adv_examples:
            logger.warning(f"Batch {batch_idx} not found in adversarial examples, skipping")
            continue
        
        # Get pre-generated adversarial examples
        adv_batch = adv_examples[batch_idx]
        image_ids = adv_batch['image_ids']
        perturbed_pixel_values = adv_batch['perturbed_pixel_values'].to(device)
        clean_similarity = adv_batch['clean_similarity'].to(device)
        
        # First evaluate without defense
        with torch.no_grad():
            # Prepare inputs based on model type
            if hasattr(model, 'get_image_features'):
                # CLIP-like model
                # Get image features for adversarial images without defense
                adv_image_embeds = model.get_image_features(pixel_values=perturbed_pixel_values)
                
                # Get/restore text embeddings
                if adv_batch['text_embeds'] is not None:
                    text_embeds = adv_batch['text_embeds'].to(device)
                else:
                    # We need to get text embeddings again
                    text_inputs = {
                        'input_ids': adv_batch['inputs']['input_ids'].to(device),
                        'attention_mask': adv_batch['inputs']['attention_mask'].to(device)
                    }
                    text_embeds = model.get_text_features(**text_inputs)
                
                # Normalize embeddings
                adv_image_embeds = adv_image_embeds / adv_image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate adversarial similarity scores
                adv_similarity = torch.matmul(adv_image_embeds, text_embeds.t())
            else:
                # Other model types - prepare full input dictionary
                adv_inputs = {}
                for key, value in adv_batch['inputs'].items():
                    if isinstance(value, torch.Tensor):
                        adv_inputs[key] = value.to(device)
                    else:
                        adv_inputs[key] = value
                
                # Override pixel values with adversarial ones
                if 'pixel_values' in adv_inputs:
                    adv_inputs['pixel_values'] = perturbed_pixel_values
                else:
                    adv_inputs['images'] = perturbed_pixel_values
                
                # Forward pass with adversarial images (no defense)
                adv_outputs = model(**adv_inputs)
                
                # Extract similarity based on model type
                if hasattr(adv_outputs, 'logits_per_image'):
                    adv_similarity = adv_outputs.logits_per_image
                elif hasattr(adv_outputs, 'similarity_scores'):
                    adv_similarity = adv_outputs.similarity_scores
                else:
                    # Fallback for other output types
                    if hasattr(adv_outputs, 'image_embeds') and hasattr(adv_outputs, 'text_embeds'):
                        image_embeds = adv_outputs.image_embeds
                        text_embeds = adv_outputs.text_embeds
                        
                        # Normalize embeddings
                        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarity
                        adv_similarity = torch.matmul(image_embeds, text_embeds.t())
                    else:
                        logger.error("Could not calculate similarity scores from model outputs")
                        adv_similarity = torch.zeros_like(clean_similarity)
        
        # Now evaluate with defense (if provided)
        if defense_instance is not None:
            # Process adversarial images with defense
            with torch.no_grad():
                if hasattr(model, 'get_image_features'):
                    # CLIP-like model with defense applied
                    defended_image_embeds = model.get_image_features(pixel_values=perturbed_pixel_values)
                    
                    # Normalize embeddings
                    defended_image_embeds = defended_image_embeds / defended_image_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate defended similarity scores
                    defended_similarity = torch.matmul(defended_image_embeds, text_embeds.t())
                else:
                    # Other model types
                    defended_outputs = model(**adv_inputs)
                    
                    # Extract similarity based on model type
                    if hasattr(defended_outputs, 'logits_per_image'):
                        defended_similarity = defended_outputs.logits_per_image
                    elif hasattr(defended_outputs, 'similarity_scores'):
                        defended_similarity = defended_outputs.similarity_scores
                    else:
                        # Fallback
                        if hasattr(defended_outputs, 'image_embeds') and hasattr(defended_outputs, 'text_embeds'):
                            image_embeds = defended_outputs.image_embeds
                            text_embeds = defended_outputs.text_embeds
                            
                            # Normalize embeddings
                            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                            
                            # Calculate similarity
                            defended_similarity = torch.matmul(image_embeds, text_embeds.t())
                        else:
                            logger.error("Could not calculate similarity scores from model outputs")
                            defended_similarity = torch.zeros_like(clean_similarity)
        else:
            # No defense applied - use same results
            defended_similarity = adv_similarity
        
        # Store image IDs
        metrics['image_ids'].extend(image_ids)
        
        # Calculate retrieval metrics for this batch
        batch_size = len(image_ids)
        targets = torch.arange(batch_size).to(device)
        
        # Calculate recall@k for clean, adversarial, and defended
        batch_result = {'batch': batch_idx, 'image_ids': image_ids}
        
        for k in [1, 5, 10]:
            if batch_size >= k:
                clean_recall = calculate_recall_at_k(clean_similarity, targets, k)
                adv_recall = calculate_recall_at_k(adv_similarity, targets, k)
                defended_recall = calculate_recall_at_k(defended_similarity, targets, k)
                
                all_recalls_clean[k].append(clean_recall)
                all_recalls_adv[k].append(adv_recall)
                all_recalls_defended[k].append(defended_recall)
                
                # Store batch metrics
                batch_result[f'clean_recall_at_{k}'] = clean_recall
                batch_result[f'adv_recall_at_{k}'] = adv_recall
                batch_result[f'defended_recall_at_{k}'] = defended_recall
                
                # Calculate improvements
                improvement = defended_recall - adv_recall
                batch_result[f'improvement_at_{k}'] = improvement
                
                # Calculate recovery
                if clean_recall > adv_recall:
                    recovery = (defended_recall - adv_recall) / (clean_recall - adv_recall) * 100
                    batch_result[f'recovery_percent_at_{k}'] = recovery
                else:
                    batch_result[f'recovery_percent_at_{k}'] = 0
        
        batch_metrics.append(batch_result)
        batch_count += 1
        
        # Free up memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Remove the hooks after evaluation if defense was applied
    if defense_instance is not None and hasattr(defense_instance, 'remove_hooks'):
        defense_instance.remove_hooks()
    
    # Calculate average recall across all batches
    for k in [1, 5, 10]:
        if all_recalls_clean[k]:
            metrics[f'clean_recall_at_{k}'] = np.mean(all_recalls_clean[k])
            metrics[f'adv_recall_at_{k}'] = np.mean(all_recalls_adv[k])
            metrics[f'defended_recall_at_{k}'] = np.mean(all_recalls_defended[k])
    
    # Calculate performance improvements and recovery
    for k in [1, 5, 10]:
        if f'clean_recall_at_{k}' in metrics:
            clean_recall = metrics[f'clean_recall_at_{k}']
            adv_recall = metrics[f'adv_recall_at_{k}']
            defended_recall = metrics[f'defended_recall_at_{k}']
            
            # Improvement over no defense
            metrics[f'improvement_at_{k}'] = defended_recall - adv_recall
            if adv_recall > 0:
                metrics[f'improvement_percent_at_{k}'] = (defended_recall - adv_recall) / adv_recall * 100
            else:
                metrics[f'improvement_percent_at_{k}'] = 0
            
            # Drop from clean performance
            metrics[f'clean_to_adv_drop_at_{k}'] = clean_recall - adv_recall
            metrics[f'clean_to_defended_drop_at_{k}'] = clean_recall - defended_recall
            
            # Recovery percentage
            attack_drop = clean_recall - adv_recall
            if attack_drop > 0:
                metrics[f'recovery_percent_at_{k}'] = (defended_recall - adv_recall) / attack_drop * 100
            else:
                metrics[f'recovery_percent_at_{k}'] = 0
    
    # Print summary results
    logger.info(f"\n{defense_name} Evaluation Results (averaged over {batch_count} batches):")
    for k in [1, 5, 10]:
        if f'clean_recall_at_{k}' in metrics:
            clean_recall = metrics[f'clean_recall_at_{k}']
            adv_recall = metrics[f'adv_recall_at_{k}']
            defended_recall = metrics[f'defended_recall_at_{k}']
            improvement_percent = metrics[f'improvement_percent_at_{k}']
            recovery_percent = metrics[f'recovery_percent_at_{k}']
            
            logger.info(f"  Recall@{k}:")
            logger.info(f"    Clean: {clean_recall:.4f}")
            logger.info(f"    Adversarial (No Defense): {adv_recall:.4f}")
            logger.info(f"    Adversarial (With {defense_name}): {defended_recall:.4f}")
            logger.info(f"    Improvement: {improvement_percent:.2f}%")
            logger.info(f"    Performance Recovery: {recovery_percent:.2f}% of the adversarial drop")
    
    return metrics, batch_metrics