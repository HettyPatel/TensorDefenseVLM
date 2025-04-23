# TensorDefenseVLM: Tensor Decomposition Defense for CLIP Models

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

TensorDefenseVLM is a defense framework that uses tensor decomposition techniques to protect CLIP models against adversarial attacks. The framework implements various tensor decomposition methods (CP, Tucker, and Tensor Train) to defend against adversarial perturbations in the intermediate representations of CLIP.

## Features

- **Multiple Decomposition Methods**:
  - Canonical Polyadic (CP) Decomposition
  - Tucker Decomposition
  - Tensor Train (TT) Decomposition

- **Flexible Defense Configuration**:
  - Single-layer and multi-layer defenses
  - Configurable rank and alpha parameters
  - Support for different target layers (final_norm, attention, mlp)

- **Supported Models**:
  - CLIP (ViT-Base-Patch32)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TensorDefenseVLM.git
cd TensorDefenseVLM
```

2. Install dependencies:
```bash
pip install -e .
```

## Usage Examples

### Basic Single-Layer Defense

Here's how to set up a basic experiment with CP decomposition on the final normalization layer:

```yaml
# configs/basic_cp.yaml
model:
  name: "clip"
  variant: "vit-base-patch32"

dataset:
  name: "nlphuji/flickr30k"
  split: "test"
  max_samples: 1000

attack:
  epsilon: 0.0313725  # 8/255
  steps: 10
  step_size: 0.0235294  # 6/255

defenses:
  - name: "CP Final Norm"
    method: "cp"
    rank: 32
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1
```

Run the experiment:
```bash
python src/experiment.py --config configs/basic_cp.yaml
```

### Multi-Layer Defense

Example of defending multiple layers using Tensor Train decomposition:

```yaml
# configs/multi_layer_tt.yaml
model:
  name: "clip"
  variant: "vit-base-patch32"

dataset:
  name: "nlphuji/flickr30k"
  split: "test"
  max_samples: 1000

attack:
  epsilon: 0.0313725
  steps: 10
  step_size: 0.0235294

defenses:
  - name: "TT Multi-Layer (3 Layers)"
    layers:
      - method: "tt"
        rank: 32
        alpha: 0.2
        target_layer: "final_norm"
        vision_layer_idx: -1
      - method: "tt"
        rank: 32
        alpha: 0.2
        target_layer: "final_norm"
        vision_layer_idx: -2
      - method: "tt"
        rank: 32
        alpha: 0.2
        target_layer: "final_norm"
        vision_layer_idx: -3
```

### Comparing Different Decomposition Methods

Example configuration to compare different decomposition methods:

```yaml
# configs/method_comparison.yaml
model:
  name: "clip"
  variant: "vit-base-patch32"

dataset:
  name: "nlphuji/flickr30k"
  split: "test"
  max_samples: 1000

attack:
  epsilon: 0.0313725
  steps: 10
  step_size: 0.0235294

defenses:
  - name: "CP Final Norm"
    method: "cp"
    rank: 32
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1

  - name: "Tucker Final Norm"
    method: "tucker"
    rank: 32
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1

  - name: "TT Final Norm"
    method: "tt"
    rank: 32
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1
```

### Parameter Sensitivity Analysis

Example to analyze the effect of different ranks and alpha values:

```yaml
# configs/parameter_sensitivity.yaml
model:
  name: "clip"
  variant: "vit-base-patch32"

dataset:
  name: "nlphuji/flickr30k"
  split: "test"
  max_samples: 1000

attack:
  epsilon: 0.0313725
  steps: 10
  step_size: 0.0235294

defenses:
  - name: "CP Rank 16"
    method: "cp"
    rank: 16
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1

  - name: "CP Rank 32"
    method: "cp"
    rank: 32
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1

  - name: "CP Rank 64"
    method: "cp"
    rank: 64
    alpha: 0.2
    target_layer: "final_norm"
    vision_layer_idx: -1

  - name: "CP Alpha 0.1"
    method: "cp"
    rank: 32
    alpha: 0.1
    target_layer: "final_norm"
    vision_layer_idx: -1

  - name: "CP Alpha 0.5"
    method: "cp"
    rank: 32
    alpha: 0.5
    target_layer: "final_norm"
    vision_layer_idx: -1
```

## Project Structure

```
TensorDefenseVLM/
├── configs/             # Experiment configurations
├── src/
│   ├── attacks/        # Adversarial attack implementations
│   ├── defenses/       # Tensor decomposition defense implementations
│   ├── models/         # Model loading and processing
│   ├── utils/          # Utility functions and metrics
│   └── experiment.py   # Main experiment runner
├── results/            # Experiment results and visualizations
└── setup.py           # Package installation configuration
```

## Results

The framework generates comprehensive results including:
- Defense effectiveness metrics (recall@1, recall@5, recall@10)
- Sample images showing defense effects
- Parameter sensitivity analysis
- Layer-wise performance comparison

Results are saved in the `results` directory with the following structure:
```
results/
├── experiment_name/
│   ├── figures/        # Generated plots and visualizations
│   ├── samples/        # Sample images showing defense effects
│   ├── metrics/        # Detailed metrics in CSV format
│   └── results.json    # Complete experiment results
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourcitation,
  title={TensorDefenseVLM: Tensor Decomposition Defense for Vision-Language Models},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue or contact the maintainers. 