from setuptools import setup, find_packages

setup(
    name="tensor_defense_vlm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.25.0",
        "tensorly>=0.7.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "pandas>=1.4.0",
        "pyyaml>=6.0",
        "datasets>=2.8.0",
        "Pillow>=9.3.0",
    ],
    author="Het Patel",
    author_email="hpate061@ucr.edu",
    description="Tensor Decomposition Defense for Vision-Language Models",
)