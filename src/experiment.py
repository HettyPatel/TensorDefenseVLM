"""
Simple experiment runner for tensor decomposition defense
"""

import os
import torch
import logging
import argparse
import yaml
import time
import json
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.models.model_loader import load_model
from src.datasets.dataset_wrapper import HFDatasetWrapper, custom_collate_fn
from src.attacks.pgd import PGDAttack
from src.defenses.tensor_defense import TargetedTensorDefense
from src.defenses.multi_layer import MultiLayerTensorDefense
from src.utils.metrics import calculate_metrics, print_metrics_summary
from src.utils.visualization import save_sample_images, create_recall_comparison_plot, create_all_paper_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tensor_defense")

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    
def run_experiment(config_path):
    """
    Run a tensor decomposition defense experiment
    
    Args:
        config_path: Path to experiment configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract experiment name from config path or use timestamp
    if isinstance(config_path, str):
        experiment_name = os.path.basename(config_path).replace('.yaml', '')
    else:
        experiment_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment-specific results directory
    base_results_dir = config.get('results_dir', 'results')
    results_dir = os.path.join(base_results_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    figures_dir = os.path.join(results_dir, 'figures')
    samples_dir = os.path.join(results_dir, 'samples')
    metrics_dir = os.path.join(results_dir, 'metrics')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save config to the experiment directory
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Load model
    model_config = config['model']
    model_name = model_config['name']
    model_variant = model_config['variant']
    
    logger.info(f"Loading model: {model_name} {model_variant}")
    model, processor = load_model(model_name, model_variant, device)
    
    # Load dataset
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    split = dataset_config['split']
    max_samples = dataset_config.get('max_samples', 100)
    
    logger.info(f"Loading dataset: {dataset_name}, split: {split}, max_samples: {max_samples}")
    
    hf_dataset = load_dataset(dataset_name)
    dataset = HFDatasetWrapper(hf_dataset, split=split, max_samples=max_samples)
    
    # Use single batch size for all operations
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    # Configure attack
    attack_config = config['attack']
    epsilon = attack_config.get('epsilon', 8/255)
    steps = attack_config.get('steps', 2)
    step_size = attack_config.get('step_size', 6/255)
    
    logger.info(f"Configuring PGD attack: epsilon={epsilon}, steps={steps}, step_size={step_size}")
    attack = PGDAttack(
        model, processor,
        epsilon=epsilon,
        alpha=step_size,
        steps=steps
    )
    
    # Initialize results storage
    all_results = {
        'clean': {'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0},
        'adversarial': {'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0}
    }
    
    # Initialize defense results
    for defense in config['defenses']:
        all_results[defense['name']] = {'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0}
    
    # Process each batch
    logger.info("Processing batches...")
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
        
        images = batch['image']
        captions = batch['caption']
        
        # Generate adversarial examples
        adv_images, inputs, orig_images = attack.perturb(images, captions, device)
        
        # Save sample images from first batch
        if batch_idx == 0:
            save_sample_images(
                orig_images, 
                adv_images, 
                captions,
                max_samples=5,
                epsilon=epsilon,
                steps=steps,
                step_size=step_size,
                save_dir=samples_dir  # Use experiment-specific samples directory
            )
        
        # Calculate similarity matrices
        with torch.no_grad():
            # For CLIP-like models
            if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
                # Get text features
                inputs = processor(
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
                
                text_features = model.get_text_features(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Get clean image features
                clean_image_features = model.get_image_features(pixel_values=orig_images)
                clean_image_features = clean_image_features / clean_image_features.norm(dim=-1, keepdim=True)
                
                # Get adversarial image features (no defense)
                adv_image_features = model.get_image_features(pixel_values=adv_images)
                adv_image_features = adv_image_features / adv_image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity matrices
                clean_similarity = torch.matmul(clean_image_features, text_features.t())
                adv_no_defense_similarity = torch.matmul(adv_image_features, text_features.t())
                
                # Calculate metrics for each defense
                batch_results = {}
                batch_results['clean'] = calculate_metrics(
                    clean_similarity, clean_similarity, clean_similarity
                )
                batch_results['adversarial'] = calculate_metrics(
                    clean_similarity, adv_no_defense_similarity, adv_no_defense_similarity
                )
                
                for defense in config['defenses']:
                    defense_name = defense['name']
                    method = defense.get('method', 'cp')
                    rank = defense.get('rank', 64)
                    alpha = defense.get('alpha', 0.5)
                    target_layer = defense.get('target_layer', 'final_norm')
                    vision_layer_idx = defense.get('vision_layer_idx', -1)
                    
                    # Apply defense
                    if isinstance(defense.get('layers', None), list):
                        defense_model = MultiLayerTensorDefense(model, defense['layers'])
                    else:
                        defense_model = TargetedTensorDefense(
                            model=model,
                            method=method,
                            rank=rank,
                            alpha=alpha,
                            target_layer=target_layer,
                            vision_layer_idx=vision_layer_idx
                        )
                    
                    # Get defended image features
                    defended_image_features = model.get_image_features(pixel_values=adv_images)
                    defended_image_features = defended_image_features / defended_image_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate defended similarity
                    defended_similarity = torch.matmul(defended_image_features, text_features.t())
                    
                    # Calculate metrics
                    batch_results[defense_name] = calculate_metrics(
                        clean_similarity, adv_no_defense_similarity, defended_similarity
                    )
                    
                    # Remove hooks
                    defense_model.remove_hooks()
            
            else:
                # For BLIP-like models
                inputs = processor(
                    text=captions,
                    images=orig_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
                
                clean_outputs = model(**inputs)
                if hasattr(clean_outputs, 'logits_per_image'):
                    clean_similarity = clean_outputs.logits_per_image
                else:
                    clean_similarity = clean_outputs.similarity_scores
                
                # Get adversarial outputs
                adv_inputs = inputs.copy()
                adv_inputs['pixel_values'] = adv_images
                adv_outputs = model(**adv_inputs)
                if hasattr(adv_outputs, 'logits_per_image'):
                    adv_no_defense_similarity = adv_outputs.logits_per_image
                else:
                    adv_no_defense_similarity = adv_outputs.similarity_scores
                
                # Calculate metrics for each defense
                batch_results = {}
                batch_results['clean'] = calculate_metrics(
                    clean_similarity, clean_similarity, clean_similarity
                )
                batch_results['adversarial'] = calculate_metrics(
                    clean_similarity, adv_no_defense_similarity, adv_no_defense_similarity
                )
                
                for defense in config['defenses']:
                    defense_name = defense['name']
                    method = defense.get('method', 'cp')
                    rank = defense.get('rank', 64)
                    alpha = defense.get('alpha', 0.5)
                    target_layer = defense.get('target_layer', 'final_norm')
                    vision_layer_idx = defense.get('vision_layer_idx', -1)
                    
                    # Apply defense
                    if isinstance(defense.get('layers', None), list):
                        defense_model = MultiLayerTensorDefense(model, defense['layers'])
                    else:
                        defense_model = TargetedTensorDefense(
                            model=model,
                            method=method,
                            rank=rank,
                            alpha=alpha,
                            target_layer=target_layer,
                            vision_layer_idx=vision_layer_idx
                        )
                    
                    # Get defended outputs
                    defended_outputs = model(**adv_inputs)
                    if hasattr(defended_outputs, 'logits_per_image'):
                        defended_similarity = defended_outputs.logits_per_image
                    else:
                        defended_similarity = defended_outputs.similarity_scores
                    
                    # Calculate metrics
                    batch_results[defense_name] = calculate_metrics(
                        clean_similarity, adv_no_defense_similarity, defended_similarity
                    )
                    
                    # Remove hooks
                    defense_model.remove_hooks()
        
        # Accumulate results
        for key in all_results:
            for k in [1, 5, 10]:
                if key == 'clean':
                    metric = f'clean_recall@{k}'
                elif key == 'adversarial':
                    metric = f'adversarial_recall@{k}'
                else:
                    metric = f'defended_recall@{k}'
                
                if metric in batch_results[key]:
                    all_results[key][f'recall@{k}'] += batch_results[key][metric]
        
        # Clear GPU memory
        del adv_images, orig_images, inputs
        torch.cuda.empty_cache()
    
    # Average results over all batches
    for key in all_results:
        for metric in ['recall@1', 'recall@5', 'recall@10']:
            all_results[key][metric] /= total_batches
    
    # Print final metrics
    print_metrics_summary(all_results)
    
    # Create standard comparison plot (current code)
    comparison_plot = create_recall_comparison_plot(all_results)
    comparison_plot.savefig(os.path.join(figures_dir, 'defense_comparison.png'))

    # Generate all paper plots
    visualize_experiment_results(all_results, config, figures_dir)  # Pass figures_dir instead of results_dir

    # Save metrics to CSV in the metrics directory
    metrics_df = pd.DataFrame()
    for defense_name, defense_results in all_results.items():
        row = {
            'defense': defense_name,
            'clean_recall@1': defense_results['recall@1'],
            'clean_recall@5': defense_results['recall@5'],
            'clean_recall@10': defense_results['recall@10'],
            'adversarial_recall@1': defense_results['recall@1'],
            'adversarial_recall@5': defense_results['recall@5'],
            'adversarial_recall@10': defense_results['recall@10'],
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
    
    metrics_df.to_csv(os.path.join(metrics_dir, 'metrics.csv'), index=False)
    
    # Save full results as JSON
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Experiment {experiment_name} completed. Results saved to {results_dir}")
    
    return results_dir, all_results

def visualize_experiment_results(all_results, config, results_dir):
    """Create all visualization plots for experiment results"""
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Format for visualization functions
    plot_format = {
        'clean': {
            'recall@1': all_results['clean']['recall@1'],
            'recall@5': all_results['clean']['recall@5'],
            'recall@10': all_results['clean']['recall@10']
        },
        'adversarial': {
            'recall@1': all_results['adversarial']['recall@1'],
            'recall@5': all_results['adversarial']['recall@5'],
            'recall@10': all_results['adversarial']['recall@10']
        }
    }
    
    # Add defense results
    for defense_name, defense_results in all_results.items():
        if defense_name not in ['clean', 'adversarial']:
            plot_format[defense_name] = {
                'recall@1': defense_results['recall@1'],
                'recall@5': defense_results['recall@5'], 
                'recall@10': defense_results['recall@10']
            }
    
    # Extract parameter sensitivity data
    param_results = {}
    
    # Group by rank (looking for defenses with same method, alpha, layer but different ranks)
    rank_defenses = {}
    for defense in config['defenses']:
        if 'layers' not in defense:  # Skip multi-layer defenses
            key = (defense.get('method', 'cp'), 
                   defense.get('alpha', 0.5), 
                   defense.get('target_layer', 'final_norm'))
            rank = defense.get('rank', 64)
            if key not in rank_defenses:
                rank_defenses[key] = []
            rank_defenses[key].append((rank, defense['name']))
    
    # Find a consistent set for rank analysis 
    for key, defenses in rank_defenses.items():
        if len(defenses) >= 3:  # Need at least 3 points for a good plot
            method, alpha, target_layer = key
            # This is a valid rank parameter sweep
            rank_results = {}
            for rank, name in defenses:
                if name in all_results:
                    rank_results[rank] = {
                        'recall@1': all_results[name]['recall@1'],
                        'recall@5': all_results[name]['recall@5'],
                        'recall@10': all_results[name]['recall@10']
                    }
            if rank_results:
                param_results['rank'] = rank_results
                break
    
    # Group by alpha (looking for defenses with same method, rank, layer but different alphas)
    alpha_defenses = {}
    for defense in config['defenses']:
        if 'layers' not in defense:  # Skip multi-layer defenses
            key = (defense.get('method', 'cp'), 
                   defense.get('rank', 64), 
                   defense.get('target_layer', 'final_norm'))
            alpha = defense.get('alpha', 0.5)
            if key not in alpha_defenses:
                alpha_defenses[key] = []
            alpha_defenses[key].append((alpha, defense['name']))
    
    # Find a consistent set for alpha analysis
    for key, defenses in alpha_defenses.items():
        if len(defenses) >= 3:  # Need at least 3 points for a good plot
            method, rank, target_layer = key
            # This is a valid alpha parameter sweep
            alpha_results = {}
            for alpha, name in defenses:
                if name in all_results:
                    alpha_results[alpha] = {
                        'recall@1': all_results[name]['recall@1'],
                        'recall@5': all_results[name]['recall@5'],
                        'recall@10': all_results[name]['recall@10']
                    }
            if alpha_results:
                param_results['alpha'] = alpha_results
                break
    
    # Group by layer type (looking for defenses with same method, rank, alpha but different layers)
    layer_defenses = {}
    for defense in config['defenses']:
        if 'layers' not in defense:  # Skip multi-layer defenses
            key = (defense.get('method', 'cp'), 
                   defense.get('rank', 64), 
                   defense.get('alpha', 0.5))
            layer = defense.get('target_layer', 'final_norm')
            if key not in layer_defenses:
                layer_defenses[key] = {}
            layer_defenses[key][layer] = defense['name']
    
    # Find a consistent set for layer analysis
    layer_results = None
    for key, layers in layer_defenses.items():
        if len(layers) >= 2:  # Need at least 2 different layers
            # This is a valid layer analysis set
            method, rank, alpha = key
            temp_results = {}
            for layer_name, defense_name in layers.items():
                if defense_name in all_results:
                    temp_results[layer_name] = {
                        'recall@1': all_results[defense_name]['recall@1'],
                        'recall@5': all_results[defense_name]['recall@5'],
                        'recall@10': all_results[defense_name]['recall@10']
                    }
            if temp_results:
                layer_results = temp_results
                break
    
    # Generate all plots
    create_all_paper_plots(plot_format, param_results, layer_results, save_dir=figures_dir)
    logger.info(f"All visualization plots saved to {figures_dir}")
    
    return figures_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tensor decomposition defense experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment configuration file')
    
    args = parser.parse_args()
    run_experiment(args.config)