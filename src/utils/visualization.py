"""
Visualization utilities for the tensor decomposition defense project.

This module provides functions for generating plots and visualizations
to understand the effectiveness of different defense configurations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import logging

logger = logging.getLogger("tensor_defense")

def save_sample_images(original_images, adversarial_images, captions, 
                      save_dir, indices=None, max_samples=5, epsilon=8/255, steps=2, step_size=6/255):
    """
    Save original and adversarial image pairs for visualization
    
    Args:
        original_images: Tensor of original image pixel values
        adversarial_images: Tensor of adversarial image pixel values
        captions: List of corresponding image captions
        save_dir: Directory to save visualization images
        indices: Specific indices to save (default: first max_samples)
        max_samples: Maximum number of samples to save
        epsilon: Perturbation magnitude used in attack
        steps: Number of attack steps
        step_size: Step size used in attack
    """
    # Create directory if it doesn't exist
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
            
            # Also save as individual PNGs for inclusion in reports
            # Convert tensors to PIL images - ensure proper range [0,255]
            orig_img_pil = Image.fromarray((orig_img * 255).astype(np.uint8))
            adv_img_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
            
            # Save individual images
            orig_img_pil.save(os.path.join(save_dir, f"original_{i+1}.png"))
            adv_img_pil.save(os.path.join(save_dir, f"adversarial_{i+1}.png"))
            
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


def create_defense_comparison_plot(all_results, k=1, save_path=None):
    """
    Create bar chart comparing defense techniques
    
    Args:
        all_results: List of result dictionaries from evaluate_defense
        k: k value for Recall@k to compare
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = "defense_comparison.pdf"
    
    # Extract data from results
    defense_names = [result['defense_name'] for result in all_results]
    clean_recalls = [result[f'clean_recall_at_{k}'] * 100 for result in all_results]
    adv_recalls = [result[f'adv_recall_at_{k}'] * 100 for result in all_results]
    defended_recalls = [result[f'defended_recall_at_{k}'] * 100 for result in all_results]
    
    # Sort by defended recall
    sorted_indices = np.argsort(defended_recalls)[::-1]  # Descending order
    defense_names = [defense_names[i] for i in sorted_indices]
    clean_recalls = [clean_recalls[i] for i in sorted_indices]
    adv_recalls = [adv_recalls[i] for i in sorted_indices]
    defended_recalls = [defended_recalls[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(defense_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, clean_recalls, width=barWidth, label='Clean', color='#2ca02c')
    plt.bar(r2, adv_recalls, width=barWidth, label='Adversarial', color='#d62728')
    plt.bar(r3, defended_recalls, width=barWidth, label='Defended', color='#1f77b4')
    
    # Add labels
    plt.xlabel('Defense Method', fontsize=14)
    plt.ylabel(f'Recall@{k} (%)', fontsize=14)
    plt.title(f'Comparison of Defense Methods (Recall@{k})', fontsize=16)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(defense_names))], defense_names, rotation=45, ha='right')
    
    # Create legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    bars1 = plt.bar(r1, clean_recalls, width=barWidth)
    bars2 = plt.bar(r2, adv_recalls, width=barWidth)
    bars3 = plt.bar(r3, defended_recalls, width=barWidth)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_recovery_plot(all_results, save_path=None):
    """
    Create plot showing recovery percentage for different defenses
    
    Args:
        all_results: List of result dictionaries from evaluate_defense
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = "recovery_comparison.pdf"
    
    # Extract data from results
    defense_names = [result['defense_name'] for result in all_results]
    recovery_r1 = [result.get('recovery_percent_at_1', 0) for result in all_results]
    recovery_r5 = [result.get('recovery_percent_at_5', 0) for result in all_results]
    recovery_r10 = [result.get('recovery_percent_at_10', 0) for result in all_results]
    
    # Create a dataframe for easier plotting
    data = {
        'Defense': defense_names,
        'Recall@1': recovery_r1,
        'Recall@5': recovery_r5,
        'Recall@10': recovery_r10
    }
    df = pd.DataFrame(data)
    
    # Sort by Recall@1 recovery
    df = df.sort_values('Recall@1', ascending=False)
    
    # Reshape for plotting
    df_melted = df.melt(id_vars='Defense', var_name='Metric', value_name='Recovery (%)')
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    ax = sns.barplot(x='Defense', y='Recovery (%)', hue='Metric', data=df_melted)
    
    # Customize
    plt.title('Recovery Percentage by Defense Method', fontsize=16)
    plt.xlabel('Defense Method', fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='', fontsize=12)
    
    # Add a horizontal line at 0%
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_parameter_sensitivity_plot(param_results, param_name, metric='recovery_percent_at_1', save_path=None):
    """
    Create plot showing sensitivity to a specific parameter
    
    Args:
        param_results: Dictionary with parameter values as keys and result dictionaries as values
        param_name: Name of the parameter (e.g., 'rank', 'alpha')
        metric: Metric to plot
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = f"{param_name}_sensitivity.pdf"
    
    # Extract parameter values and metrics
    param_values = sorted(param_results.keys())
    metric_values = []
    
    for param_val in param_values:
        result = param_results[param_val]
        metric_values.append(result.get(metric, 0))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create line plot with markers
    plt.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
    
    # Add data labels
    for x, y in zip(param_values, metric_values):
        if 'recall' in metric:
            label = f"{y*100:.1f}%" if not isinstance(y, str) else y
        else:
            label = f"{y:.1f}%" if not isinstance(y, str) else y
        plt.annotate(label, (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    # Customize
    title_metric = metric.replace('_', ' ').replace('at', '@').title()
    plt.title(f'Sensitivity to {param_name.title()} ({title_metric})', fontsize=16)
    plt.xlabel(param_name.title(), fontsize=14)
    
    if 'recall' in metric:
        plt.ylabel('Recall (%)', fontsize=14)
        plt.ylim(0, 100)
    else:
        plt.ylabel('Recovery (%)', fontsize=14)
    
    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_results_table(all_results, save_path=None):
    """
    Create a table summarizing results across all defenses
    
    Args:
        all_results: List of result dictionaries from evaluate_defense
        save_path: Path to save the table as CSV
        
    Returns:
        DataFrame with summarized results
    """
    if save_path is None:
        save_path = "defense_results_summary.csv"
    
    # Create rows for the table
    rows = []
    
    for result in all_results:
        defense_name = result['defense_name']
        
        # Extract key metrics
        clean_r1 = result.get('clean_recall_at_1', 0) * 100
        adv_r1 = result.get('adv_recall_at_1', 0) * 100
        defended_r1 = result.get('defended_recall_at_1', 0) * 100
        improvement_r1 = result.get('improvement_at_1', 0) * 100
        recovery_r1 = result.get('recovery_percent_at_1', 0)
        
        clean_r5 = result.get('clean_recall_at_5', 0) * 100
        adv_r5 = result.get('adv_recall_at_5', 0) * 100
        defended_r5 = result.get('defended_recall_at_5', 0) * 100
        improvement_r5 = result.get('improvement_at_5', 0) * 100
        recovery_r5 = result.get('recovery_percent_at_5', 0)
        
        # Get defense parameters if available
        if isinstance(result.get('defense_config', None), dict):
            config = result['defense_config']
            method = config.get('method', 'N/A')
            rank = config.get('rank', 'N/A')
            alpha = config.get('alpha', 'N/A')
            target_layer = config.get('target_layer', 'N/A')
            layer_idx = config.get('vision_layer_idx', 'N/A')
        else:
            method = 'N/A'
            rank = 'N/A'
            alpha = 'N/A'
            target_layer = 'N/A'
            layer_idx = 'N/A'
        
        # Create row
        row = {
            'Defense': defense_name,
            'Method': method,
            'Rank': rank,
            'Alpha': alpha,
            'Target_Layer': target_layer,
            'Layer_Index': layer_idx,
            'Clean_R@1': f"{clean_r1:.2f}",
            'Adv_R@1': f"{adv_r1:.2f}",
            'Defended_R@1': f"{defended_r1:.2f}",
            'Improvement_R@1': f"{improvement_r1:.2f}",
            'Recovery_R@1': f"{recovery_r1:.2f}",
            'Clean_R@5': f"{clean_r5:.2f}",
            'Adv_R@5': f"{adv_r5:.2f}",
            'Defended_R@5': f"{defended_r5:.2f}",
            'Improvement_R@5': f"{improvement_r5:.2f}",
            'Recovery_R@5': f"{recovery_r5:.2f}"
        }
        
        rows.append(row)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    
    # Also create a formatted HTML version for reports
    if save_path.endswith('.csv'):
        html_path = save_path.replace('.csv', '.html')
        html_content = df.to_html(index=False)
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    return df