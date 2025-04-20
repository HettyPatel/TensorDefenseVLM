"""
Main entry point for tensor decomposition defense experiments
"""

import os
import argparse
import logging
import yaml
from src.experiment import run_experiment

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tensor_defense")

def create_directory_structure():
    """Create the necessary directory structure if it doesn't exist"""
    dirs = [
        "src/models",
        "src/datasets", 
        "src/attacks",
        "src/defenses",
        "src/utils",
        "results",
        "results/samples",
        "configs"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Directory structure created")

def main():
    parser = argparse.ArgumentParser(description="Run tensor decomposition defense experiment")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default=None, 
                      help='Custom experiment name (defaults to config filename)')
    parser.add_argument('--create-dirs', action='store_true', help='Create directory structure')
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_directory_structure()
    
    # For each experiment config, create a separate folder
    config_path = args.config
    experiment_name = args.experiment_name
    
    if experiment_name:
        # Read config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override results directory with experiment name
        config['results_dir'] = os.path.join(config.get('results_dir', 'results'), experiment_name)
        
        # Create a temporary config with the experiment name
        temp_config_path = f"temp_{os.path.basename(config_path)}"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run with temp config
        results_dir, results = run_experiment(temp_config_path)
        
        # Clean up temp file
        os.remove(temp_config_path)
    else:
        # Just use the config filename as experiment name
        results_dir, results = run_experiment(config_path)
    
    logger.info(f"Experiment completed. Results available in {results_dir}")

if __name__ == "__main__":
    main()