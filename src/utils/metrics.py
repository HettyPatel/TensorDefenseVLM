"""
Evaluation metrics for Vision-Language Models
"""

import torch
import numpy as np
import logging

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


def calculate_metrics(clean_similarity, adv_no_defense, adv_with_defense):
    """
    Calculate comprehensive metrics comparing clean, attacked, and defended performance
    
    Args:
        clean_similarity: Similarity matrix for clean images
        adv_no_defense: Similarity matrix for adversarial images without defense
        adv_with_defense: Similarity matrix for adversarial images with defense
        
    Returns:
        Dictionary of metrics
    """
    batch_size = clean_similarity.size(0)
    targets = torch.arange(batch_size).to(clean_similarity.device)
    
    metrics = {}
    
    # Calculate recall@k for k in [1, 5, 10]
    for k in [1, 5, 10]:
        if batch_size >= k:
            # Clean performance
            clean_recall = calculate_recall_at_k(clean_similarity, targets, k)
            metrics[f'clean_recall@{k}'] = clean_recall
            
            # Attacked performance (no defense)
            no_defense_recall = calculate_recall_at_k(adv_no_defense, targets, k)
            metrics[f'adversarial_recall@{k}'] = no_defense_recall
            
            # Defended performance
            defended_recall = calculate_recall_at_k(adv_with_defense, targets, k)
            metrics[f'defended_recall@{k}'] = defended_recall
            
            # Calculate improvements
            abs_improvement = defended_recall - no_defense_recall
            metrics[f'abs_improvement@{k}'] = abs_improvement
            
            rel_improvement = (abs_improvement / no_defense_recall * 100) if no_defense_recall > 0 else 0
            metrics[f'rel_improvement@{k}'] = rel_improvement
            
            # Calculate recovery percentage
            attack_drop = clean_recall - no_defense_recall
            defense_drop = clean_recall - defended_recall
            
            if attack_drop > 0:
                recovery = (attack_drop - defense_drop) / attack_drop * 100
                metrics[f'recovery_percent@{k}'] = recovery
            else:
                metrics[f'recovery_percent@{k}'] = 0
    
    return metrics


def print_metrics_summary(results):
    """
    Print a summary of the evaluation metrics
    
    Args:
        results: Dictionary containing metrics for each defense
    """
    logger = logging.getLogger("tensor_defense")
    
    logger.info("\nDefense Evaluation Results:")
    logger.info("-" * 50)
    
    for defense_name, metrics in results.items():
        logger.info(f"\n{defense_name}:")
        logger.info(f"  Recall@1: {metrics['recall@1']:.4f}")
        logger.info(f"  Recall@5: {metrics['recall@5']:.4f}")
        logger.info(f"  Recall@10: {metrics['recall@10']:.4f}")
    
    logger.info("-" * 50)