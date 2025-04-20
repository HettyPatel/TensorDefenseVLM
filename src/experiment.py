"""
Simple experiment runner for tensor decomposition defense
"""

import os
import torch
import logging
import argparse
import yaml
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
    
    # Create results directory
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'samples'), exist_ok=True)
    
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
                save_dir=os.path.join(results_dir, 'samples')
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
    
    # Create and save comparison plot
    comparison_plot = create_recall_comparison_plot(all_results)
    comparison_plot.savefig(os.path.join(results_dir, 'defense_comparison.png'))
    
    logger.info(f"Experiment completed. Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tensor decomposition defense experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment configuration file')
    
    args = parser.parse_args()
    run_experiment(args.config)