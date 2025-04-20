"""
Main entry point for tensor decomposition defense experiments
"""

import os
import argparse
import logging
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
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Tensor Decomposition Defense for VLMs')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to experiment configuration file')
    parser.add_argument('--create-dirs', action='store_true', help='Create directory structure')
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_directory_structure()
    
    # Run experiment with the specified configuration
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    logger.info(f"Running experiment with config: {config_path}")
    run_experiment(config_path)
    
    logger.info("Experiment completed")

if __name__ == "__main__":
    main()