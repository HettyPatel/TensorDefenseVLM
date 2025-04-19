#!/usr/bin/env python
"""
Script for running parameter sweeps for tensor decomposition defense.

This script runs multiple experiments by varying one parameter at a time
to analyze the sensitivity of defense performance to different hyperparameters.
"""

import os
import sys
import argparse
import yaml
import logging
import copy
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from run_experiment import run_experiment, load_config
from src.utils.visualization import create_parameter_sensitivity_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tensor_defense")

def run_alpha_sweep(base_config, output_dir, alpha_values=None):
    """Run experiment sweep over different alpha values"""
    if alpha_values is None:
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Create output directory
    alpha_dir = os.path.join(output_dir, "alpha_sweep")
    os.makedirs(alpha_dir, exist_ok=True)
    
    # Store results for each value
    results = {}
    
    # Run experiment for each alpha value
    for alpha in alpha_values:
        logger.info(f"Running experiment with alpha={alpha}")
        
        # Create modified config
        config = copy.deepcopy(base_config)
        
        # Set alpha value for all defenses
        for defense in config.get('defenses', []):
            defense['alpha'] = alpha
        
        # Set output directory
        config['results_dir'] = os.path.join(alpha_dir, f"alpha_{alpha}")
        
        # Run experiment
        _, experiment_results = run_experiment(config)
        
        # Store results (use the first defense result)
        for result in experiment_results:
            if result.get('defense_name') != "No Defense":
                results[alpha] = result
                break
    
    # Create sensitivity plot
    create_parameter_sensitivity_plot(
        results, 
        param_name='alpha',
        metric='recovery_percent_at_1',
        save_path=os.path.join(alpha_dir, "alpha_sensitivity_r1.pdf")
    )
    create_parameter_sensitivity_plot(
        results, 
        param_name='alpha',
        metric='recovery_percent_at_5',
        save_path=os.path.join(alpha_dir, "alpha_sensitivity_r5.pdf")
    )
    
    # Create results table
    results_df = pd.DataFrame([
        {
            'Alpha': alpha,
            'Clean_R@1': result.get('clean_recall_at_1', 0) * 100,
            'Adv_R@1': result.get('adv_recall_at_1', 0) * 100,
            'Defended_R@1': result.get('defended_recall_at_1', 0) * 100,
            'Recovery_R@1': result.get('recovery_percent_at_1', 0),
            'Clean_R@5': result.get('clean_recall_at_5', 0) * 100,
            'Adv_R@5': result.get('adv_recall_at_5', 0) * 100,
            'Defended_R@5': result.get('defended_recall_at_5', 0) * 100,
            'Recovery_R@5': result.get('recovery_percent_at_5', 0)
        }
        for alpha, result in results.items()
    ])
    
    # Sort by alpha value
    results_df = results_df.sort_values('Alpha')
    
    # Save to CSV
    results_df.to_csv(os.path.join(alpha_dir, "alpha_sweep_results.csv"), index=False)
    
    return results_df

def run_rank_sweep(base_config, output_dir, rank_values=None):
    """Run experiment sweep over different rank values"""
    if rank_values is None:
        rank_values = [16, 32, 64, 128, 256]
    
    # Create output directory
    rank_dir = os.path.join(output_dir, "rank_sweep")
    os.makedirs(rank_dir, exist_ok=True)
    
    # Store results for each value
    results = {}
    
    # Run experiment for each rank value
    for rank in rank_values:
        logger.info(f"Running experiment with rank={rank}")
        
        # Create modified config
        config = copy.deepcopy(base_config)
        
        # Set rank value for all defenses
        for defense in config.get('defenses', []):
            defense['rank'] = rank
        
        # Set output directory
        config['results_dir'] = os.path.join(rank_dir, f"rank_{rank}")
        
        # Run experiment
        _, experiment_results = run_experiment(config)
        
        # Store results (use the first defense result)
        for result in experiment_results:
            if result.get('defense_name') != "No Defense":
                results[rank] = result
                break
    
    # Create sensitivity plot
    create_parameter_sensitivity_plot(
        results, 
        param_name='rank',
        metric='recovery_percent_at_1',
        save_path=os.path.join(rank_dir, "rank_sensitivity_r1.pdf")
    )
    create_parameter_sensitivity_plot(
        results, 
        param_name='rank',
        metric='recovery_percent_at_5',
        save_path=os.path.join(rank_dir, "rank_sensitivity_r5.pdf")
    )
    
    # Create results table
    results_df = pd.DataFrame([
        {
            'Rank': rank,
            'Clean_R@1': result.get('clean_recall_at_1', 0) * 100,
            'Adv_R@1': result.get('adv_recall_at_1', 0) * 100,
            'Defended_R@1': result.get('defended_recall_at_1', 0) * 100,
            'Recovery_R@1': result.get('recovery_percent_at_1', 0),
            'Clean_R@5': result.get('clean_recall_at_5', 0) * 100,
            'Adv_R@5': result.get('adv_recall_at_5', 0) * 100,
            'Defended_R@5': result.get('defended_recall_at_5', 0) * 100,
            'Recovery_R@5': result.get('recovery_percent_at_5', 0)
        }
        for rank, result in results.items()
    ])
    
    # Sort by rank value
    results_df = results_df.sort_values('Rank')
    
    # Save to CSV
    results_df.to_csv(os.path.join(rank_dir, "rank_sweep_results.csv"), index=False)
    
    return results_df

def run_layer_sweep(base_config, output_dir, layer_indices=None):
    """Run experiment sweep over different layer indices"""
    if layer_indices is None:
        layer_indices = [-1, -2, -3, -4, -5]
    
    # Create output directory
    layer_dir = os.path.join(output_dir, "layer_sweep")
    os.makedirs(layer_dir, exist_ok=True)
    
    # Store results for each value
    results = {}
    
    # Run experiment for each layer index
    for layer_idx in layer_indices:
        logger.info(f"Running experiment with layer_idx={layer_idx}")
        
        # Create modified config
        config = copy.deepcopy(base_config)
        
        # Set layer index value for all defenses
        for defense in config.get('defenses', []):
            defense['vision_layer_idx'] = layer_idx
        
        # Set output directory
        config['results_dir'] = os.path.join(layer_dir, f"layer_{layer_idx}")
        
        # Run experiment
        _, experiment_results = run_experiment(config)
        
        # Store results (use the first defense result)
        for result in experiment_results:
            if result.get('defense_name') != "No Defense":
                results[layer_idx] = result
                break
    
    # Create sensitivity plot
    create_parameter_sensitivity_plot(
        results, 
        param_name='layer_index',
        metric='recovery_percent_at_1',
        save_path=os.path.join(layer_dir, "layer_sensitivity_r1.pdf")
    )
    create_parameter_sensitivity_plot(
        results, 
        param_name='layer_index',
        metric='recovery_percent_at_5',
        save_path=os.path.join(layer_dir, "layer_sensitivity_r5.pdf")
    )
    
    # Create results table
    results_df = pd.DataFrame([
        {
            'Layer_Index': layer_idx,
            'Clean_R@1': result.get('clean_recall_at_1', 0) * 100,
            'Adv_R@1': result.get('adv_recall_at_1', 0) * 100,
            'Defended_R@1': result.get('defended_recall_at_1', 0) * 100,
            'Recovery_R@1': result.get('recovery_percent_at_1', 0),
            'Clean_R@5': result.get('clean_recall_at_5', 0) * 100,
            'Adv_R@5': result.get('adv_recall_at_5', 0) * 100,
            'Defended_R@5': result.get('defended_recall_at_5', 0) * 100,
            'Recovery_R@5': result.get('recovery_percent_at_5', 0)
        }
        for layer_idx, result in results.items()
    ])
    
    # Sort by layer index value
    results_df = results_df.sort_values('Layer_Index')
    
    # Save to CSV
    results_df.to_csv(os.path.join(layer_dir, "layer_sweep_results.csv"), index=False)
    
    return results_df

def run_method_sweep(base_config, output_dir, methods=None):
    """Run experiment sweep over different decomposition methods"""
    if methods is None:
        methods = ['cp', 'tucker', 'tt']
    
    # Create output directory
    method_dir = os.path.join(output_dir, "method_sweep")
    os.makedirs(method_dir, exist_ok=True)
    
    # Store results for each value
    results = {}
    
    # Run experiment for each method
    for method in methods:
        logger.info(f"Running experiment with method={method}")
        
        # Create modified config
        config = copy.deepcopy(base_config)
        
        # Set method value for all defenses
        for defense in config.get('defenses', []):
            defense['method'] = method
        
        # Set output directory
        config['results_dir'] = os.path.join(method_dir, f"method_{method}")
        
        # Run experiment
        _, experiment_results = run_experiment(config)
        
        # Store results (use the first defense result)
        for result in experiment_results:
            if result.get('defense_name') != "No Defense":
                results[method] = result
                break
    
    # Create results table
    results_df = pd.DataFrame([
        {
            'Method': method,
            'Clean_R@1': result.get('clean_recall_at_1', 0) * 100,
            'Adv_R@1': result.get('adv_recall_at_1', 0) * 100,
            'Defended_R@1': result.get('defended_recall_at_1', 0) * 100,
            'Recovery_R@1': result.get('recovery_percent_at_1', 0),
            'Clean_R@5': result.get('clean_recall_at_5', 0) * 100,
            'Adv_R@5': result.get('adv_recall_at_5', 0) * 100,
            'Defended_R@5': result.get('defended_recall_at_5', 0) * 100,
            'Recovery_R@5': result.get('recovery_percent_at_5', 0)
        }
        for method, result in results.items()
    ])
    
    # Save to CSV
    results_df.to_csv(os.path.join(method_dir, "method_sweep_results.csv"), index=False)
    
    # Create bar chart for methods
    plt.figure(figsize=(12, 6))
    
    x = results_df['Method']
    y1 = results_df['Recovery_R@1']
    y5 = results_df['Recovery_R@5']
    
    bar_width = 0.35
    index = range(len(x))
    
    plt.bar([i - bar_width/2 for i in index], y1, width=bar_width, label='Recall@1')
    plt.bar([i + bar_width/2 for i in index], y5, width=bar_width, label='Recall@5')
    
    plt.xlabel('Decomposition Method')
    plt.ylabel('Recovery Percentage (%)')
    plt.title('Recovery Percentage by Decomposition Method')
    plt.xticks(index, x)
    plt.legend()
    
    plt.savefig(os.path.join(method_dir, "method_comparison.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(method_dir, "method_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description="Run parameter sweeps for tensor decomposition defense")
    parser.add_argument("--config", type=str, required=True, help="Path to base configuration YAML file")
    parser.add_argument("--param", choices=['alpha', 'rank', 'layer', 'method', 'all'], 
                      default='all', help="Parameter to sweep over")
    parser.add_argument("--alpha-values", type=float, nargs='+', 
                      help="Alpha values to test (default: [0.1, 0.3, 0.5, 0.7, 0.9])")
    parser.add_argument("--rank-values", type=int, nargs='+', 
                      help="Rank values to test (default: [16, 32, 64, 128, 256])")
    parser.add_argument("--layer-indices", type=int, nargs='+', 
                      help="Layer indices to test (default: [-1, -2, -3, -4, -5])")
    parser.add_argument("--methods", type=str, nargs='+', 
                      help="Methods to test (default: ['cp', 'tucker', 'tt'])")
    
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Create timestamp for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up output directory
    output_dir = f"parameter_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base configuration
    with open(os.path.join(output_dir, "base_config.yaml"), 'w') as f:
        yaml.dump(base_config, f)
    
    # Run requested parameter sweeps
    results = {}
    
    if args.param == 'alpha' or args.param == 'all':
        logger.info("Running alpha parameter sweep")
        results['alpha'] = run_alpha_sweep(base_config, output_dir, args.alpha_values)
    
    if args.param == 'rank' or args.param == 'all':
        logger.info("Running rank parameter sweep")
        results['rank'] = run_rank_sweep(base_config, output_dir, args.rank_values)
    
    if args.param == 'layer' or args.param == 'all':
        logger.info("Running layer index parameter sweep")
        results['layer'] = run_layer_sweep(base_config, output_dir, args.layer_indices)
    
    if args.param == 'method' or args.param == 'all':
        logger.info("Running decomposition method sweep")
        results['method'] = run_method_sweep(base_config, output_dir, args.methods)
    
    logger.info(f"Parameter sweep completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()