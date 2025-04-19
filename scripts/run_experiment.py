#!/usr/bin/env python
"""
Main script for running tensor decomposition defense experiments.

This script provides a configurable way to run experiments testing the
effectiveness of tensor decomposition defenses against adversarial attacks
on vision-language models like CLIP and LLaVA.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.attacks.pgd import PGDAttack
from src.attacks.fgsm import FGSM
from src.defenses.tensor_defense import TargetedTensorDefense
from src.defenses.multilayer_defense import MultiLayerTensorDefense
from src.datasets.data_loader import get_dataloader
from src.utils.evaluation import generate_adversarial_examples, evaluate_defense
from src.utils.visualization import (
    save_sample_images, 
    create_defense_comparison_plot,
    create_recovery_plot, 
    create_parameter_sensitivity_plot,
    create_results_table
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tensor_defense")

def load_config(config_path):
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_config, device):
    """Load model based on configuration"""
    model_type = model_config.get('type', 'clip')
    model_name = model_config.get('name')
    
    if model_type.lower() == 'clip':
        from transformers import CLIPModel, CLIPProcessor
        
        logger.info(f"Loading CLIP model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        
    elif model_type.lower() == 'llava':
        try:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            
            logger.info(f"Loading LLaVA model: {model_name}")
            model = LlavaForConditionalGeneration.from_pretrained(model_name).to(device)
            processor = LlavaProcessor.from_pretrained(model_name)
        except ImportError:
            logger.error("Could not import LLaVA-related classes. Make sure transformers is updated.")
            raise
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, processor

def get_attack(attack_config, model, processor):
    """Create attack based on configuration"""
    attack_type = attack_config.get('type', 'pgd')
    
    if attack_type.lower() == 'pgd':
        attack = PGDAttack(
            model=model,
            processor=processor,
            epsilon=attack_config.get('epsilon', 8/255),
            alpha=attack_config.get('step_size', 6/255),
            steps=attack_config.get('steps', 2),
            random_start=attack_config.get('random_start', True)
        )
    elif attack_type.lower() == 'fgsm':
        attack = FGSM(
            model=model,
            processor=processor,
            epsilon=attack_config.get('epsilon', 8/255)
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    return attack

def get_defense(defense_config, model):
    """Create defense based on configuration"""
    if defense_config is None:
        return None
    
    defense_type = defense_config.get('type', 'targeted')
    
    if defense_type.lower() == 'targeted':
        defense = TargetedTensorDefense(
            model=model,
            method=defense_config.get('method', 'cp'),
            rank=defense_config.get('rank', 64),
            alpha=defense_config.get('alpha', 0.5),
            target_layer=defense_config.get('target_layer', 'final_norm'),
            vision_layer_idx=defense_config.get('vision_layer_idx', -1)
        )
    elif defense_type.lower() == 'multilayer':
        layer_configs = defense_config.get('layer_configs', [])
        defense = MultiLayerTensorDefense(
            model=model,
            layer_configs=layer_configs
        )
    else:
        raise ValueError(f"Unsupported defense type: {defense_type}")
    
    return defense

def run_experiment(config_path):
    """Run experiment based on configuration"""
    # Load configuration
    config = load_config(config_path)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up results directory
    results_dir = config.get('results_dir', 'results')
    results_dir = f"results/{results_dir}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    samples_dir = os.path.join(results_dir, "sample_images")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load model
    model_config = config.get('model', {})
    model, processor = load_model(model_config, device)
    
    # Load dataset
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'nlphuji/flickr30k')
    split = dataset_config.get('split', 'test')
    max_samples = dataset_config.get('max_samples', 1000)
    batch_size = dataset_config.get('batch_size', 32)
    
    logger.info(f"Loading dataset: {dataset_name}, split: {split}, max_samples: {max_samples}")
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        max_samples=max_samples,
        num_workers=dataset_config.get('num_workers', 4)
    )
    
    # Create attack
    attack_config = config.get('attack', {})
    attack = get_attack(attack_config, model, processor)
    
    # Generate adversarial examples
    max_batches = config.get('max_batches', None)
    logger.info("Generating adversarial examples")
    adv_examples, sample_data = generate_adversarial_examples(
        model=model, 
        processor=processor, 
        dataloader=dataloader, 
        attack=attack, 
        device=device,
        max_batches=max_batches
    )
    
    # Save sample images
    sample_originals, sample_adversarials, sample_captions = sample_data
    if sample_originals is not None:
        logger.info("Saving sample images")
        save_sample_images(
            original_images=sample_originals,
            adversarial_images=sample_adversarials,
            captions=sample_captions,
            save_dir=samples_dir,
            epsilon=attack_config.get('epsilon', 8/255),
            steps=attack_config.get('steps', 2),
            step_size=attack_config.get('step_size', 6/255)
        )
    
    # Evaluate with different defenses
    defense_configs = config.get('defenses', [])
    all_results = []
    
    # First evaluate with no defense as baseline
    logger.info("Evaluating baseline (no defense)")
    baseline_metrics, baseline_batch_metrics = evaluate_defense(
        model=model,
        processor=processor,
        dataloader=dataloader,
        adv_examples=adv_examples,
        defense=None,
        defense_name="No Defense",
        device=device,
        max_batches=max_batches
    )
    
    # Add baseline to results
    all_results.append(baseline_metrics)
    
    # Save baseline batch metrics
    pd.DataFrame(baseline_batch_metrics).to_csv(
        os.path.join(results_dir, "no_defense_batch_metrics.csv"),
        index=False
    )
    
    # Evaluate each defense configuration
    for defense_config in defense_configs:
        defense_name = defense_config.get('name', 'Unnamed Defense')
        logger.info(f"Evaluating defense: {defense_name}")
        
        # Create defense
        defense = get_defense(defense_config, model)
        
        # Evaluate defense
        defense_metrics, defense_batch_metrics = evaluate_defense(
            model=model,
            processor=processor,
            dataloader=dataloader,
            adv_examples=adv_examples,
            defense=defense,
            defense_name=defense_name,
            device=device,
            max_batches=max_batches
        )
        
        # Add defense configuration to metrics
        defense_metrics['defense_config'] = defense_config
        all_results.append(defense_metrics)
        
        # Save defense batch metrics
        defense_file_name = defense_name.lower().replace(' ', '_').replace('(','').replace(')','')
        pd.DataFrame(defense_batch_metrics).to_csv(
            os.path.join(results_dir, f"{defense_file_name}_batch_metrics.csv"),
            index=False
        )
        
        # Free up memory
        if defense is not None and hasattr(defense, 'remove_hooks'):
            defense.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create visualizations
    logger.info("Creating visualizations")
    create_defense_comparison_plot(
        all_results, 
        k=1, 
        save_path=os.path.join(plots_dir, "defense_comparison_r1.pdf")
    )
    create_defense_comparison_plot(
        all_results, 
        k=5, 
        save_path=os.path.join(plots_dir, "defense_comparison_r5.pdf")
    )
    create_recovery_plot(
        all_results,
        save_path=os.path.join(plots_dir, "recovery_comparison.pdf")
    )
    
    # Create results table
    create_results_table(
        all_results,
        save_path=os.path.join(results_dir, "defense_results_summary.csv")
    )
    
    # Save complete results
    with open(os.path.join(results_dir, "complete_results.json"), 'w') as f:
        # Convert non-serializable values (like tensors) to strings
        serializable_results = []
        for result in all_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    # Save configuration
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Experiment completed. Results saved to {results_dir}")
    return results_dir, all_results

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description="Run tensor decomposition defense experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    # Run experiment
    run_experiment(args.config)

if __name__ == "__main__":
    main()