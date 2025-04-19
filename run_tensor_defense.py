#!/usr/bin/env python
"""
Quick start script for tensor decomposition defense against adversarial attacks.

This script provides a convenient way to run experiments with different models,
datasets, attacks, and defense configurations.
"""

import os
import argparse
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tensor_defense")

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description="Run tensor decomposition defense experiments")
    
    # Model options
    parser.add_argument("--model", choices=['clip-b32', 'clip-b16', 'clip-l14', 'llava'], 
                       default='clip-b32', help="Model architecture to use")
    
    # Dataset options
    parser.add_argument("--dataset", choices=['flickr30k', 'coco'], 
                       default='flickr30k', help="Dataset to use")
    parser.add_argument("--max-samples", type=int, default=1000, 
                       help="Maximum number of samples to use")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for dataloader")
    
    # Attack options
    parser.add_argument("--attack", choices=['pgd', 'fgsm'], 
                       default='pgd', help="Attack method to use")
    parser.add_argument("--epsilon", type=float, default=8/255, 
                       help="Perturbation magnitude (default: 8/255)")
    parser.add_argument("--steps", type=int, default=2, 
                       help="Number of attack steps (for PGD)")
    
    # Defense options
    parser.add_argument("--defense", choices=['cp', 'tucker', 'tt'], 
                       default='cp', help="Tensor decomposition method")
    parser.add_argument("--rank", type=int, default=64, 
                       help="Rank for tensor decomposition")
    parser.add_argument("--alpha", type=float, default=0.5, 
                       help="Weight for residual connection")
    parser.add_argument("--layer", choices=['final_norm', 'attention', 'mlp'], 
                       default='final_norm', help="Target layer type")
    parser.add_argument("--layer-idx", type=int, default=-1, 
                       help="Index of vision encoder layer to protect")
    
    # Output options
    parser.add_argument("--results-dir", type=str, default=None, 
                       help="Directory to save results (default: results_TIMESTAMP)")
    parser.add_argument("--skip-visualizations", action='store_true', 
                       help="Skip generating visualizations to save time")
    
    args = parser.parse_args()
    
    # Generate YAML configuration based on arguments
    config = generate_config(args)
    
    # Save configuration to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or f"results/results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    config_path = os.path.join(results_dir, "config.yaml")
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    logger.info(f"Configuration saved to {config_path}")
    
    # Run experiment
    logger.info("Starting experiment")
    try:
        # Add scripts directory to Python path
        script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
        sys.path.append(script_dir)
        
        from scripts.run_experiment import run_experiment
        config_path = os.path.join(results_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        run_experiment(config_path)
        
        logger.info(f"Experiment completed. Results saved to {results_dir}")
    except ImportError:
        logger.error("Could not import run_experiment. Make sure you're in the project root directory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        sys.exit(1)

def generate_config(args):
    """Generate configuration dictionary from command-line arguments"""
    # Map model choice to model configuration
    model_map = {
        'clip-b32': {'type': 'clip', 'name': 'openai/clip-vit-base-patch32'},
        'clip-b16': {'type': 'clip', 'name': 'openai/clip-vit-base-patch16'},
        'clip-l14': {'type': 'clip', 'name': 'openai/clip-vit-large-patch14'},
        'llava': {'type': 'llava', 'name': 'llava-hf/llava-1.5-7b-hf'}
    }
    
    # Map dataset choice to dataset configuration
    dataset_map = {
        'flickr30k': {'name': 'nlphuji/flickr30k', 'split': 'test'},
        'coco': {'name': 'laion/coco', 'split': 'validation'}
    }
    
    # Generate timestamp for results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create configuration dictionary
    config = {
        'random_seed': 42,
        'results_dir': args.results_dir or f"results_{timestamp}",
        'model': model_map[args.model],
        'dataset': {
            **dataset_map[args.dataset],
            'max_samples': args.max_samples,
            'batch_size': args.batch_size,
            'num_workers': 4
        },
        'attack': {
            'type': args.attack,
            'epsilon': args.epsilon,
            'steps': args.steps,
            'step_size': args.epsilon * 0.75,  # Default step size
            'random_start': True
        },
        'defenses': [
            {
                'name': f"{args.defense.upper()} {args.layer} (R={args.rank}, alpha={args.alpha})",
                'type': 'targeted',
                'method': args.defense,
                'rank': args.rank,
                'alpha': args.alpha,
                'target_layer': args.layer,
                'vision_layer_idx': args.layer_idx
            }
        ],
        'skip_visualizations': args.skip_visualizations
    }
    
    return config

if __name__ == "__main__":
    main()