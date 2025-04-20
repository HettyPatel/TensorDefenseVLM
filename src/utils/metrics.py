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
            metrics[f'clean_recall_at_{k}'] = clean_recall
            
            # Attacked performance (no defense)
            no_defense_recall = calculate_recall_at_k(adv_no_defense, targets, k)
            metrics[f'no_defense_recall_at_{k}'] = no_defense_recall
            
            # Defended performance
            defended_recall = calculate_recall_at_k(adv_with_defense, targets, k)
            metrics[f'defended_recall_at_{k}'] = defended_recall
            
            # Calculate improvements
            abs_improvement = defended_recall - no_defense_recall
            metrics[f'abs_improvement_at_{k}'] = abs_improvement
            
            rel_improvement = (abs_improvement / no_defense_recall * 100) if no_defense_recall > 0 else 0
            metrics[f'rel_improvement_at_{k}'] = rel_improvement
            
            # Calculate recovery percentage
            attack_drop = clean_recall - no_defense_recall
            defense_drop = clean_recall - defended_recall
            
            if attack_drop > 0:
                recovery = (attack_drop - defense_drop) / attack_drop * 100
                metrics[f'recovery_percent_at_{k}'] = recovery
            else:
                metrics[f'recovery_percent_at_{k}'] = 0
    
    return metrics


def print_metrics_summary(metrics, defense_name="Defense"):
    """
    Print a summary of evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        defense_name: Name of the defense for printing
    """
    logger.info(f"\n{defense_name} Evaluation Results:")
    for k in [1, 5, 10]:
        if f'clean_recall_at_{k}' in metrics:
            clean_recall = metrics[f'clean_recall_at_{k}']
            adv_recall = metrics[f'no_defense_recall_at_{k}']
            defended_recall = metrics[f'defended_recall_at_{k}']
            improvement_percent = metrics.get(f'rel_improvement_at_{k}', 0)
            recovery_percent = metrics.get(f'recovery_percent_at_{k}', 0)
            
            logger.info(f"  Recall@{k}:")
            logger.info(f"    Clean: {clean_recall:.4f}")
            logger.info(f"    Adversarial (No Defense): {adv_recall:.4f}")
            logger.info(f"    Adversarial (With {defense_name}): {defended_recall:.4f}")
            logger.info(f"    Improvement: {improvement_percent:.2f}%")
            logger.info(f"    Performance Recovery: {recovery_percent:.2f}% of the adversarial drop")