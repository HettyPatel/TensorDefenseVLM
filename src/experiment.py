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
from src.utils.visualization import save_sample_images, create_comparison_plot

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
    
    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)
    
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
    
    # Generate adversarial examples
    logger.info("Generating adversarial examples")
    all_results = {}
    
    # Process first batch for demonstration
    batch = next(iter(dataloader))
    image_ids = batch['image_id']
    images = batch['image']
    captions = batch['caption']
    
    # Generate adversarial examples
    adv_images, inputs, orig_images = attack.perturb(images, captions, device)
    
    # Save sample images
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
    
    # Evaluate clean performance
    with torch.no_grad():
        # For CLIP-like models
        if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
            clean_image_embeds = model.get_image_features(pixel_values=orig_images)
            text_embeds = model.get_text_features(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            # Normalize embeddings
            clean_image_embeds = clean_image_embeds / clean_image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Calculate clean similarity scores
            clean_similarity = torch.matmul(clean_image_embeds, text_embeds.t())
            
        else:
            # For BLIP-like models
            clean_outputs = model(**inputs)
            if hasattr(clean_outputs, 'logits_per_image'):
                clean_similarity = clean_outputs.logits_per_image
            else:
                clean_similarity = clean_outputs.similarity_scores
    
    # Evaluate adversarial performance (no defense)
    with torch.no_grad():
        # For CLIP-like models
        if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
            adv_image_embeds = model.get_image_features(pixel_values=adv_images)
            
            # Normalize embeddings
            adv_image_embeds = adv_image_embeds / adv_image_embeds.norm(dim=-1, keepdim=True)
            
            # Calculate adversarial similarity scores
            adv_similarity = torch.matmul(adv_image_embeds, text_embeds.t())
            
        else:
            # For BLIP-like models
            adv_inputs = inputs.copy()
            adv_inputs['pixel_values'] = adv_images
            
            adv_outputs = model(**adv_inputs)
            if hasattr(adv_outputs, 'logits_per_image'):
                adv_similarity = adv_outputs.logits_per_image
            else:
                adv_similarity = adv_outputs.similarity_scores
    
    # Store baseline results
    all_results["No Defense"] = {
        'clean_similarity': clean_similarity,
        'adv_similarity': adv_similarity,
        # Just pass the same matrix twice since there's no defense
        'metrics': calculate_metrics(clean_similarity, adv_similarity, adv_similarity)
    }
    
    print_metrics_summary(all_results["No Defense"]['metrics'], "No Defense")
    
    # Test different defense configurations
    defense_configs = config.get('defenses', [])
    
    for defense_config in defense_configs:
        defense_name = defense_config['name']
        method = defense_config.get('method', 'cp')
        rank = defense_config.get('rank', 64)
        alpha = defense_config.get('alpha', 0.5)
        target_layer = defense_config.get('target_layer', 'final_norm')
        vision_layer_idx = defense_config.get('vision_layer_idx', -1)
        
        logger.info(f"Evaluating defense: {defense_name}")
        
        # Apply defense
        if isinstance(defense_config.get('layers', None), list):
            # Multi-layer defense
            defense = MultiLayerTensorDefense(model, defense_config['layers'])
        else:
            # Single-layer defense
            defense = TargetedTensorDefense(
                model=model,
                method=method,
                rank=rank,
                alpha=alpha,
                target_layer=target_layer,
                vision_layer_idx=vision_layer_idx
            )
        
        # Evaluate with defense
        with torch.no_grad():
            # For CLIP-like models
            if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
                defended_image_embeds = model.get_image_features(pixel_values=adv_images)
                
                # Normalize embeddings
                defended_image_embeds = defended_image_embeds / defended_image_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate defended similarity scores
                defended_similarity = torch.matmul(defended_image_embeds, text_embeds.t())
                
            else:
                # For BLIP-like models
                defended_outputs = model(**adv_inputs)
                if hasattr(defended_outputs, 'logits_per_image'):
                    defended_similarity = defended_outputs.logits_per_image
                else:
                    defended_similarity = defended_outputs.similarity_scores
        
        # Calculate metrics
        metrics = calculate_metrics(clean_similarity, adv_similarity, defended_similarity)
        
        # Store results
        all_results[defense_name] = {
            'clean_similarity': clean_similarity,
            'adv_similarity': adv_similarity,
            'defended_similarity': defended_similarity,
            'metrics': metrics
        }
        
        print_metrics_summary(metrics, defense_name)
        
        # Remove hooks
        defense.remove_hooks()
    
    # Create comparison plot
    create_comparison_plot(
        {name: results['metrics'] for name, results in all_results.items()},
        k=1,
        save_path=os.path.join(results_dir, 'defense_comparison_r1.png')
    )
    
    create_comparison_plot(
        {name: results['metrics'] for name, results in all_results.items()},
        k=5,
        save_path=os.path.join(results_dir, 'defense_comparison_r5.png')
    )
    
    logger.info(f"Experiment completed. Results saved to {results_dir}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tensor decomposition defense experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment configuration file')
    
    args = parser.parse_args()
    run_experiment(args.config)