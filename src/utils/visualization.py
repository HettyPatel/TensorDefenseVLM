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


def create_comparison_plot(results, k=1, save_path="defense_comparison.png"):
    """
    Create bar chart comparing defense techniques
    
    Args:
        results: Dictionary with defense names as keys and metrics as values
        k: k value for Recall@k to compare
        save_path: Path to save the figure
    """
    # Extract data from results
    defense_names = list(results.keys())
    clean_recalls = [results[name][f'clean_recall_at_{k}'] * 100 for name in defense_names]
    adv_recalls = [results[name][f'no_defense_recall_at_{k}'] * 100 for name in defense_names]
    defended_recalls = [results[name][f'defended_recall_at_{k}'] * 100 for name in defense_names]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
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
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot to {save_path}")

def create_parameter_sensitivity_plot(results, metric='recall@1', title=None):
    """
    Create a parameter sensitivity plot showing how different parameter values affect the metric.
    
    Args:
        results: Dictionary of results with parameter values as keys
        metric: Metric to plot (default: 'recall@1')
        title: Optional title for the plot
        
    Returns:
        matplotlib figure object
    """
    # Extract parameter values and metric values
    param_values = []
    metric_values = []
    for param, result in results.items():
        param_values.append(param)
        metric_values.append(result[metric])
    
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metric values with improved styling
    ax.plot(param_values, metric_values, 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            markersize=8,
            color='#2ecc71',  # Emerald green
            label=f'{metric} (%)')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title with improved formatting
    ax.set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} (%)', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title(f'Parameter Sensitivity Analysis\n{metric}', 
                    fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis labels
    if all(isinstance(x, (int, float)) for x in param_values):
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    else:
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([str(x) for x in param_values], rotation=45, ha='right')
    
    # Set y-axis limits with padding
    y_min = min(metric_values)
    y_max = max(metric_values)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - padding), min(100, y_max + padding))
    
    # Add value labels on points
    for x, y in zip(param_values, metric_values):
        ax.annotate(f'{y:.2f}%', 
                   (x, y), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=10,
                   fontweight='bold')
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_defense_comparison_plot(results, title=None):
    """
    Create a bar plot comparing clean, adversarial, and defended recall values.
    
    Args:
        results: Dictionary containing clean, adversarial, and defended recall values
        title: Optional title for the plot
        
    Returns:
        matplotlib figure object
    """
    # Extract recall values
    clean_recall = results['clean']['recall@1'] * 100
    adv_recall = results['adversarial']['recall@1'] * 100
    
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    colors = {
        'clean': '#2ecc71',      # Emerald green
        'adversarial': '#e74c3c', # Alizarin red
        'defended': '#3498db'    # Peter River blue
    }
    
    # Plot clean and adversarial bars
    bars = []
    bar_labels = ['Clean', 'Adversarial']
    bar_values = [clean_recall, adv_recall]
    bar_colors = [colors['clean'], colors['adversarial']]
    
    # Plot defended bars for each defense
    for defense_name, defense_results in results.items():
        if defense_name not in ['clean', 'adversarial']:
            defended_recall = defense_results['recall@1'] * 100
            bar_labels.append(defense_name)
            bar_values.append(defended_recall)
            bar_colors.append(colors['defended'])
    
    # Create bars with improved styling
    bars = ax.bar(range(len(bar_values)), bar_values, 
                 color=bar_colors,
                 alpha=0.8,
                 edgecolor='black',
                 linewidth=1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', 
                   va='bottom',
                   fontsize=10,
                   fontweight='bold')
    
    # Set labels and title with improved formatting
    ax.set_xlabel('Defense Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('Defense Performance Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits with padding
    y_min = min(bar_values)
    y_max = max(bar_values)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - padding), min(100, y_max + padding))
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc=colors['clean']),
        plt.Rectangle((0,0), 1, 1, fc=colors['adversarial']),
        plt.Rectangle((0,0), 1, 1, fc=colors['defended'])
    ]
    ax.legend(legend_elements, ['Clean', 'Adversarial', 'Defended'],
             loc='upper right',
             fontsize=10)
    
    # Add horizontal line at clean performance for reference
    ax.axhline(y=clean_recall, color='gray', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig