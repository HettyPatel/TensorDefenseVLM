"""
Computational Efficiency Analysis For Tensor Decomposition Defenses
Using Real Model and Flickr30k Dataset

This script analyzes the computational efficiency of different tensor decomposition
defense methods (no defense, CP, Tucker, TT) on a real CLIP model with Flickr30k images.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse

# Import your project's modules
from src.models.model_loader import load_model
from src.custom_datasets.dataset_wrapper import HFDatasetWrapper, custom_collate_fn
from src.defenses.tensor_defense import TargetedTensorDefense

def setup_environment():
    """Set up the environment, create directories, set device"""
    # Create results directory
    os.makedirs("results/efficiency_analysis", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    
    return device

def load_dataset_and_model(device, batch_size=32, max_samples=1000):
    """Load the Flickr30k dataset and CLIP model"""
    print("Loading CLIP model...")
    model, processor = load_model("clip", "vit-base-patch32", device)
    
    print(f"Loading Flickr30k dataset (max_samples={max_samples})...")
    hf_dataset = load_dataset("nlphuji/flickr30k")
    dataset = HFDatasetWrapper(hf_dataset, split="test", max_samples=max_samples)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    return model, processor, dataloader

def measure_baseline_efficiency(model, processor, dataloader, device, num_batches=10):
    """Measure baseline efficiency (no defense)"""
    print("Measuring baseline efficiency (no defense)...")
    
    total_time = 0
    total_samples = 0
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx >= num_batches:
            break
            
        images = batch['image']
        captions = batch['caption']
        
        # Process inputs
        inputs = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Measure time for forward pass
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()  # Make sure GPU operations are completed
        execution_time = time.time() - start_time
        
        total_time += execution_time
        total_samples += len(images)
    
    avg_time_per_batch = total_time / min(num_batches, len(dataloader))
    avg_time_per_sample = total_time / total_samples
    
    result = {
        'method': 'No Defense',
        'rank': 0,
        'alpha': 0,
        'layers': 0,
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'avg_time_per_sample': avg_time_per_sample,
        'throughput': total_samples / total_time
    }
    
    print(f"Baseline: {avg_time_per_batch*1000:.2f} ms/batch, {avg_time_per_sample*1000:.2f} ms/sample")
    
    return result

def measure_defense_efficiency(model, processor, dataloader, device, 
                              method='cp', rank=64, alpha=0.5, target_layer='final_norm',
                              num_batches=10):
    """Measure efficiency with a specific defense method"""
    print(f"Measuring efficiency with {method.upper()} defense (rank={rank}, alpha={alpha})...")
    
    total_time = 0
    total_samples = 0
    
    # Initialize defense
    defense = TargetedTensorDefense(
        model=model,
        method=method,
        rank=rank,
        alpha=alpha,
        target_layer=target_layer,
        vision_layer_idx=-1
    )
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx >= num_batches:
            break
            
        images = batch['image']
        captions = batch['caption']
        
        # Process inputs
        inputs = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Measure time for forward pass with defense
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()  # Make sure GPU operations are completed
        execution_time = time.time() - start_time
        
        total_time += execution_time
        total_samples += len(images)
    
    # Remove hooks
    defense.remove_hooks()
    
    avg_time_per_batch = total_time / min(num_batches, len(dataloader))
    avg_time_per_sample = total_time / total_samples
    
    result = {
        'method': method,
        'rank': rank,
        'alpha': alpha,
        'layers': 1,
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'avg_time_per_sample': avg_time_per_sample,
        'throughput': total_samples / total_time
    }
    
    print(f"{method.upper()}: {avg_time_per_batch*1000:.2f} ms/batch, {avg_time_per_sample*1000:.2f} ms/sample")
    
    return result

def measure_multi_layer_efficiency(model, processor, dataloader, device, 
                                  method='cp', rank=64, alpha=0.5, num_layers=5,
                                  num_batches=10):
    """Measure efficiency with multi-layer defense"""
    print(f"Measuring efficiency with {method.upper()} multi-layer defense (layers={num_layers})...")
    
    total_time = 0
    total_samples = 0
    
    # Initialize multi-layer defense configuration
    layer_configs = []
    for i in range(num_layers):
        layer_configs.append({
            'method': method,
            'rank': rank,
            'alpha': alpha,
            'target_layer': 'final_norm',
            'vision_layer_idx': -(i+1)  # Last layer, second-to-last, etc.
        })
    
    
    from src.defenses.multi_layer import MultiLayerTensorDefense
    
    # Initialize defense
    defense = MultiLayerTensorDefense(model, layer_configs)
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx >= num_batches:
            break
            
        images = batch['image']
        captions = batch['caption']
        
        # Process inputs
        inputs = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Measure time for forward pass with defense
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()  # Make sure GPU operations are completed
        execution_time = time.time() - start_time
        
        total_time += execution_time
        total_samples += len(images)
    
    # Remove hooks
    defense.remove_hooks()
    
    avg_time_per_batch = total_time / min(num_batches, len(dataloader))
    avg_time_per_sample = total_time / total_samples
    
    result = {
        'method': f"{method}-multi",
        'rank': rank,
        'alpha': alpha,
        'layers': num_layers,
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'avg_time_per_sample': avg_time_per_sample,
        'throughput': total_samples / total_time
    }
    
    print(f"{method.upper()} Multi-Layer: {avg_time_per_batch*1000:.2f} ms/batch, {avg_time_per_sample*1000:.2f} ms/sample")
    
    return result

def create_efficiency_plots(results):
    """Create and save plots visualizing the efficiency results"""
    import matplotlib.pyplot as plt
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs("results/efficiency_analysis", exist_ok=True)
    
    # Extract data
    methods = [r['method'] for r in results]  # This should be 6 items
    times_per_batch = [r['avg_time_per_batch'] * 1000 for r in results]  # Convert to ms
    times_per_sample = [r['avg_time_per_sample'] * 1000 for r in results]  # Convert to ms
    throughputs = [r['throughput'] for r in results]
    
    # Find baseline for overhead calculation
    baseline_time = next((r['avg_time_per_batch'] for r in results if r['method'] == 'No Defense'), None)
    if baseline_time:
        overheads = [r['avg_time_per_batch'] / baseline_time for r in results]
    else:
        overheads = [1.0] * len(methods)
    
    # 1. Execution Time Comparison
    plt.figure(figsize=(12, 6))
    
    # Use distinct colors - make sure we have enough for all methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Create bar chart - use the same number of positions as methods
    x_positions = range(len(methods))
    bars = plt.bar(x_positions, times_per_batch, color=colors[:len(methods)])
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(x_positions, methods, rotation=45, ha='right')
    plt.title('Average Execution Time per Batch', fontsize=14, fontweight='bold')
    plt.ylabel('Time (ms)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/efficiency_analysis/execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Computational Overhead
    plt.figure(figsize=(12, 6))
    
    # Create bar chart for overhead
    bars = plt.bar(x_positions, overheads, color=colors[:len(methods)])
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(x_positions, methods, rotation=45, ha='right')
    plt.title('Computational Overhead Relative to No Defense', fontsize=14, fontweight='bold')
    plt.ylabel('Overhead Factor (×)', fontsize=12)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/efficiency_analysis/computational_overhead.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Throughput Comparison
    plt.figure(figsize=(12, 6))
    
    # Create bar chart for throughput
    bars = plt.bar(x_positions, throughputs, color=colors[:len(methods)])
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f} img/s',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(x_positions, methods, rotation=45, ha='right')
    plt.title('Throughput (Images per Second)', fontsize=14, fontweight='bold')
    plt.ylabel('Images/second', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/efficiency_analysis/throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Combined plot - overhead vs layers
    multi_layer_results = [r for r in results if r.get('layers', 0) > 0]
    if len(multi_layer_results) > 1:
        plt.figure(figsize=(10, 6))
        
        # Group by method
        method_types = list(set([r['method'].split('-')[0] if '-' in r['method'] else r['method'] for r in multi_layer_results]))
        for method in method_types:
            # Get results for this method type (either exact match or starts with method name)
            method_results = [r for r in multi_layer_results if 
                             r['method'] == method or 
                             (isinstance(r['method'], str) and r['method'].startswith(method))]
            
            if method_results:
                # Sort by number of layers
                method_results.sort(key=lambda x: x.get('layers', 1))
                layers = [r.get('layers', 1) for r in method_results]
                overheads = [r['avg_time_per_batch'] / baseline_time for r in method_results]
                
                plt.plot(layers, overheads, marker='o', linewidth=2, label=method.upper())
        
        plt.title('Overhead vs Number of Layers', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Layers', fontsize=12)
        plt.ylabel('Overhead Factor (×)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('results/efficiency_analysis/overhead_vs_layers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Combined plot - all metrics
    plt.figure(figsize=(15, 10))
    
    # 5.1 Execution Time
    plt.subplot(2, 2, 1)
    bars = plt.bar(x_positions, times_per_batch, color=colors[:len(methods)])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontsize=8)
    plt.xticks(x_positions, methods, rotation=45, ha='right', fontsize=8)
    plt.title('Execution Time (ms/batch)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 5.2 Overhead
    plt.subplot(2, 2, 2)
    bars = plt.bar(x_positions, overheads, color=colors[:len(methods)])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=8)
    plt.xticks(x_positions, methods, rotation=45, ha='right', fontsize=8)
    plt.title('Computational Overhead', fontsize=12)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 5.3 Throughput
    plt.subplot(2, 2, 3)
    bars = plt.bar(x_positions, throughputs, color=colors[:len(methods)])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)
    plt.xticks(x_positions, methods, rotation=45, ha='right', fontsize=8)
    plt.title('Throughput (images/second)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 5.4 Per-sample time
    plt.subplot(2, 2, 4)
    # Calculate per-sample times if not directly provided
    if 'avg_time_per_sample' in results[0]:
        times_per_sample = [r['avg_time_per_sample'] * 1000 for r in results]  # Convert to ms
    else:
        # If times_per_sample wasn't calculated, use batch time divided by batch size
        batch_size = 32  # Default, adjust if needed
        times_per_sample = [t / batch_size for t in times_per_batch]
        
    bars = plt.bar(x_positions, times_per_sample, color=colors[:len(methods)])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontsize=8)
    plt.xticks(x_positions, methods, rotation=45, ha='right', fontsize=8)
    plt.title('Time per Sample (ms)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/efficiency_analysis/all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All plots saved to results/efficiency_analysis/")

def main():
    """Main function to run the computational efficiency analysis"""
    parser = argparse.ArgumentParser(description="Run computational efficiency analysis on real model")
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of samples to use')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches to process for timing')
    args = parser.parse_args()
    
    print("Starting computational efficiency analysis...")
    print(f"Configuration: batch_size={args.batch_size}, max_samples={args.max_samples}, num_batches={args.num_batches}")
    
    # Setup environment
    device = setup_environment()
    
    # Load dataset and model
    model, processor, dataloader = load_dataset_and_model(device, args.batch_size, args.max_samples)
    
    # Run efficiency measurements
    results = []
    
    # 1. Baseline (no defense)
    baseline_result = measure_baseline_efficiency(model, processor, dataloader, device, args.num_batches)
    results.append(baseline_result)
    
    # 2. Single-layer defenses with different methods
    cp_result = measure_defense_efficiency(model, processor, dataloader, device, 
                                         method='cp', rank=64, alpha=0.3, 
                                         num_batches=args.num_batches)
    results.append(cp_result)
    
    tucker_result = measure_defense_efficiency(model, processor, dataloader, device, 
                                             method='tucker', rank=64, alpha=0.3, 
                                             num_batches=args.num_batches)
    results.append(tucker_result)
    
    tt_result = measure_defense_efficiency(model, processor, dataloader, device, 
                                         method='tt', rank=64, alpha=0.3, 
                                         num_batches=args.num_batches)
    results.append(tt_result)
    
    # 3. Multi-layer defenses
    cp_multi_2 = measure_multi_layer_efficiency(model, processor, dataloader, device, 
                                             method='cp', rank=64, alpha=0.3, num_layers=2,
                                             num_batches=args.num_batches)
    results.append(cp_multi_2)
    
    cp_multi_5 = measure_multi_layer_efficiency(model, processor, dataloader, device, 
                                             method='cp', rank=64, alpha=0.3, num_layers=5,
                                             num_batches=args.num_batches)
    
    tt_multi_2 = measure_multi_layer_efficiency(model, processor, dataloader, device, 
                                             method='tt', rank=64, alpha=0.3, num_layers=2,
                                             num_batches=args.num_batches)
    results.append(tt_multi_2)
    
    tt_multi_5 = measure_multi_layer_efficiency(model, processor, dataloader, device, 
                                             method='tt', rank=64, alpha=0.3, num_layers=5,
                                             num_batches=args.num_batches)
    results.append(tt_multi_5)
    
    # Create result dataframe
    df = pd.DataFrame(results)
    print("\nResults summary:")
    print(df[['method', 'rank', 'layers', 'avg_time_per_batch', 'throughput']])
    
    # Save results
    df.to_csv("results/efficiency_analysis/efficiency_results.csv", index=False)
    
    # Create plots
    create_efficiency_plots(results)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()