"""
Visualization utilities for tensor decomposition defense experiments
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

logger = logging.getLogger("tensor_defense")

def save_sample_images(original_images, adversarial_images, captions, 
                     indices=None, max_samples=5, epsilon=8/255, steps=2, step_size=6/255,
                     save_dir="sample_images"):
    """
    Save original and adversarial image pairs for visualization
    
    Args:
        original_images: Tensor of original image pixel values
        adversarial_images: Tensor of adversarial image pixel values
        captions: List of corresponding image captions
        indices: Specific indices to save (default: first max_samples)
        max_samples: Maximum number of samples to save
        epsilon: Perturbation magnitude used in attack
        steps: Number of attack steps
        step_size: Step size used in attack
        save_dir: Directory to save images
    """
    # Create directory for samples if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # If no indices provided, use first max_samples
    if indices is None:
        indices = list(range(min(max_samples, len(original_images))))
    else:
        # Limit to max_samples
        indices = indices[:max_samples]
    
    # Save each pair of original and adversarial images
    for i, idx in enumerate(indices):
        try:
            # Get original and adversarial images
            orig_img = original_images[idx].permute(1, 2, 0).cpu().numpy()
            adv_img = adversarial_images[idx].permute(1, 2, 0).cpu().numpy()
            
            # Fix image data range issues - clip to [0,1] before visualization
            orig_img = np.clip(orig_img, 0, 1)
            adv_img = np.clip(adv_img, 0, 1)
            
            # Get caption
            caption = captions[idx]
            if len(caption) > 50:
                caption = caption[:47] + "..."
            
            # Create figure with subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot original image
            axs[0].imshow(orig_img)
            axs[0].set_title("Original Image", fontsize=14)
            axs[0].axis('off')
            
            # Plot adversarial image
            axs[1].imshow(adv_img)
            axs[1].set_title(f"Adversarial Image\nε={epsilon:.4f}, steps={steps}", fontsize=14)
            axs[1].axis('off')
            
            # Plot perturbation (difference)
            perturbation = np.abs(adv_img - orig_img)
            # Normalize for better visualization
            perturbation = perturbation / perturbation.max() if perturbation.max() > 0 else perturbation
            
            axs[2].imshow(perturbation, cmap='viridis')
            axs[2].set_title("Perturbation Visualization\n(Scaled for visibility)", fontsize=14)
            axs[2].axis('off')
            
            # Add caption as suptitle
            plt.suptitle(f"Caption: {caption}", fontsize=16)
            
            # Add attack parameters text
            attack_params = (
                f"Attack Parameters:\n"
                f"Epsilon: {epsilon:.4f}, Steps: {steps}, Step Size: {step_size:.4f}"
            )
            fig.text(0.5, 0.01, attack_params, ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.1)
            
            # Save figure
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved sample pair {i+1}/{len(indices)}")
            
        except Exception as e:
            logger.error(f"Error saving sample {i+1}: {str(e)}")
    
    # Save attack parameters in a text file
    with open(os.path.join(save_dir, "attack_parameters.txt"), 'w') as f:
        f.write(f"Attack Parameters:\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Step Size: {step_size}\n")
    
    logger.info(f"Saved {len(indices)} sample image pairs to {save_dir}")


def create_recall_comparison_plot(results, k=1):
    """Create bar chart comparing defense techniques for a specific k value"""
    defense_names = [name for name in results.keys() if name not in ['clean', 'adversarial']]
    
    clean_baseline = results['clean'][f'recall@{k}'] * 100
    adv_baseline = results['adversarial'][f'recall@{k}'] * 100
    defended_recalls = [results[name][f'recall@{k}'] * 100 for name in defense_names]
    
    plt.figure(figsize=(15, 6))
    x = np.arange(len(defense_names))
    width = 0.25
    
    plt.axhline(y=clean_baseline, color='#2ca02c', linestyle='--', alpha=0.5, label='Clean Baseline')
    plt.axhline(y=adv_baseline, color='#d62728', linestyle='--', alpha=0.5, label='Adversarial Baseline')
    plt.bar(x, defended_recalls, width, label='Defended', color='#1f77b4')
    
    for i, v in enumerate(defended_recalls):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.text(-0.5, clean_baseline + 1, f'Clean: {clean_baseline:.1f}%', ha='right', va='bottom', color='#2ca02c')
    plt.text(-0.5, adv_baseline + 1, f'Adversarial: {adv_baseline:.1f}%', ha='right', va='bottom', color='#d62728')
    
    plt.xlabel('Defense Method', fontsize=12)
    plt.ylabel(f'Recall@{k} (%)', fontsize=12)
    plt.title(f'Defense Performance Comparison (Recall@{k})', fontsize=14)
    plt.xticks(x, defense_names, rotation=45, ha='right')
    plt.ylim(0, max(clean_baseline, max(defended_recalls)) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    return plt.gcf()

def create_all_recall_plots(results, save_dir):
    """Create and save recall plots for k=1,5,10"""
    os.makedirs(save_dir, exist_ok=True)
    
    for k in [1, 5, 10]:
        fig = create_recall_comparison_plot(results, k)
        fig.savefig(os.path.join(save_dir, f'recall_{k}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_recovery_plot(results):
    """Create plot showing recovery percentage for each defense"""
    defense_names = [name for name in results.keys() if name not in ['clean', 'adversarial']]
    recovery_values = []
    
    for name in defense_names:
        clean_recall = results['clean']['recall@1']
        adv_recall = results['adversarial']['recall@1']
        def_recall = results[name]['recall@1']
        
        attack_drop = clean_recall - adv_recall
        if attack_drop > 0:
            recovery = (def_recall - adv_recall) / attack_drop * 100
        else:
            recovery = 0
        recovery_values.append(recovery)
    
    plt.figure(figsize=(15, 6))
    x = np.arange(len(defense_names))
    plt.bar(x, recovery_values, color='#1f77b4')
    
    for i, v in enumerate(recovery_values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Defense Method', fontsize=12)
    plt.ylabel('Recovery Percentage (%)', fontsize=12)
    plt.title('Performance Recovery by Defense Method', fontsize=14)
    plt.xticks(x, defense_names, rotation=45, ha='right')
    plt.ylim(0, max(recovery_values) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def create_parameter_sensitivity_plot(param_results, param_name='rank'):
    """Create plot showing sensitivity to a parameter (rank or alpha)"""
    param_values = sorted(param_results.keys())
    recall_values = [param_results[param]['recall@1'] * 100 for param in param_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, recall_values, marker='o', linewidth=2, markersize=8)
    
    for x, y in zip(param_values, recall_values):
        plt.text(x, y + 1, f'{y:.1f}%', ha='center', va='bottom')
    
    plt.xlabel(f'{param_name.title()} Value', fontsize=12)
    plt.ylabel('Recall@1 (%)', fontsize=12)
    plt.title(f'Sensitivity to {param_name.title()} Parameter', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def create_layer_analysis_plot(layer_results):
    """Create plot comparing performance across different layers"""
    layer_names = list(layer_results.keys())
    recall_values = [layer_results[layer]['recall@1'] * 100 for layer in layer_names]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(layer_names))
    plt.bar(x, recall_values, color='#1f77b4')
    
    for i, v in enumerate(recall_values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Recall@1 (%)', fontsize=12)
    plt.title('Performance Across Different Layers', fontsize=14)
    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def create_all_paper_plots(results, param_results=None, layer_results=None, save_dir='paper_figures'):
    """Create and save all plots needed for the paper"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create recall plots for different k values
    create_all_recall_plots(results, save_dir)
    
    # Create recovery percentage plot
    recovery_fig = create_recovery_plot(results)
    recovery_fig.savefig(os.path.join(save_dir, 'recovery_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(recovery_fig)
    
    # Create parameter sensitivity plots if results provided
    if param_results:
        if 'rank' in param_results:
            rank_fig = create_parameter_sensitivity_plot(param_results['rank'], 'rank')
            rank_fig.savefig(os.path.join(save_dir, 'rank_sensitivity.png'), dpi=300, bbox_inches='tight')
            plt.close(rank_fig)
        
        if 'alpha' in param_results:
            alpha_fig = create_parameter_sensitivity_plot(param_results['alpha'], 'alpha')
            alpha_fig.savefig(os.path.join(save_dir, 'alpha_sensitivity.png'), dpi=300, bbox_inches='tight')
            plt.close(alpha_fig)
    
    # Create layer analysis plot if results provided
    if layer_results:
        layer_fig = create_layer_analysis_plot(layer_results)
        layer_fig.savefig(os.path.join(save_dir, 'layer_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close(layer_fig)
    
    logger.info(f"All paper figures saved to {save_dir}")