"""
Enhanced Tensor Decomposition Defense Against Adversarial Attacks on Vision-Language Models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import json
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval
from datasets import load_dataset
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train
import argparse
import yaml
from pathlib import Path
import logging
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import random
from typing import List, Dict, Tuple, Optional, Union

# Create timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories for results and samples
RESULTS_DIR = f"results_{TIMESTAMP}"
SAMPLES_DIR = os.path.join(RESULTS_DIR, "sample_images")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, f'experiment.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tensor_defense")

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# Set tensor backend
tl.set_backend('pytorch')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

#################################################
# Enhanced Tensor Decomposition Defense Classes #
#################################################

class BaseTensorDefense:
    """Base class for tensor decomposition defense strategies"""
    
    def __init__(self, model):
        """Initialize the base defense class
        
        Args:
            model: The model to defend
        """
        self.model = model
        self.hook_handles = []
        
        # Set tensorly backend
        tl.set_backend('pytorch')
    
    def remove_hooks(self):
        """Remove all registered forward hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.info("All hooks removed")
    
    def __del__(self):
        """Cleanup when object is deleted"""
        self.remove_hooks()


class TargetedTensorDefense(BaseTensorDefense):
    """Targeted tensor decomposition defense for Vision-Language Models
    
    This class applies tensor decomposition to specific layers of a CLIP model
    to defend against adversarial attacks on images. By decomposing and reconstructing
    intermediate representations, the defense aims to remove adversarial perturbations
    while preserving semantic content.
    
    Defense Mechanism:
    1. Forward hooks intercept the output of a targeted layer in the CLIP vision encoder
    2. The layer's output tensor is then decomposed using low-rank tensor decomposition:
       - CP decomposition: Represents the tensor as a sum of rank-1 tensors
       - Tucker decomposition: Represents the tensor as a core tensor multiplied by matrices
       - Tensor Train decomposition: Represents the tensor as a series of contracted core tensors
    3. The decomposed tensor is reconstructed, which removes high-frequency perturbations
       while preserving the core semantic information
    4. A residual connection combines the original output (alpha) with the 
       reconstructed output (1-alpha), controlling the defense strength
    """
    def __init__(self, model, method='cp', rank=64, alpha=0.5, 
                 target_layer='final_norm', vision_layer_idx=-1):
        """
        Initialize the tensor decomposition defense
        
        Args:
            model: CLIP/BLIP model to defend
            method: 'cp', 'tucker', or 'tt' (tensor train) decomposition
            rank: Rank for tensor decomposition
            alpha: Weight for residual connection (original * alpha + decomposed * (1-alpha))
            target_layer: Type of layer to target - 'final_norm', 'attention', 'mlp'
            vision_layer_idx: Index of vision encoder layer to protect (-1 = last layer)
        """
        super().__init__(model)
        self.method = method
        self.rank = rank
        self.alpha = alpha
        self.target_layer = target_layer
        self.vision_layer_idx = vision_layer_idx
        
        # Register hooks based on the targeted layer
        self._register_hooks()
        
        # Log the configuration
        logger.info(f"Initialized {method} tensor defense on {target_layer} (vision layer {vision_layer_idx})")
        logger.info(f"Rank: {rank}, Alpha: {alpha}")
    
    def _register_hooks(self):
        """Register forward hooks on the targeted layer"""
        # Determine model type and structure
        if hasattr(self.model, 'vision_model'):
            # CLIP model
            self._register_clip_hooks()
        elif hasattr(self.model, 'vision_encoder'):
            # BLIP model
            self._register_blip_hooks()
        else:
            logger.error(f"Unsupported model type: {type(self.model)}")
    
    def _register_clip_hooks(self):
        """Register hooks for CLIP model architecture"""
        # Get total number of vision layers for negative indexing
        num_vision_layers = len(self.model.vision_model.encoder.layers)
        
        # Convert negative index to positive if needed
        layer_idx = self.vision_layer_idx
        if layer_idx < 0:
            layer_idx = num_vision_layers + layer_idx
        
        # Ensure layer index is valid
        if layer_idx < 0 or layer_idx >= num_vision_layers:
            logger.error(f"Invalid vision layer index: {self.vision_layer_idx}")
            return
        
        # Target the specified layer type
        if self.target_layer == 'final_norm':
            try:
                # Final layer norm after the MLP
                vision_module = self.model.vision_model.encoder.layers[layer_idx].layer_norm2
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on CLIP vision encoder layer {layer_idx} final norm")
            except (AttributeError, IndexError) as e:
                logger.error(f"Could not find CLIP vision encoder final norm in layer {layer_idx}: {str(e)}")
        
        elif self.target_layer == 'attention':
            try:
                # Self-attention output
                vision_module = self.model.vision_model.encoder.layers[layer_idx].self_attn.out_proj
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on CLIP vision encoder layer {layer_idx} attention output")
            except (AttributeError, IndexError) as e:
                logger.error(f"Could not find CLIP vision encoder attention in layer {layer_idx}: {str(e)}")
        
        elif self.target_layer == 'mlp':
            try:
                # MLP output (fc2)
                vision_module = self.model.vision_model.encoder.layers[layer_idx].mlp.fc2
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on CLIP vision encoder layer {layer_idx} MLP output")
            except (AttributeError, IndexError) as e:
                logger.error(f"Could not find CLIP vision encoder MLP in layer {layer_idx}: {str(e)}")
    
    def _register_blip_hooks(self):
        """Register hooks for BLIP model architecture"""
        # Get total number of vision layers for negative indexing
        num_vision_layers = len(self.model.vision_encoder.encoder.layer)
        
        # Convert negative index to positive if needed
        layer_idx = self.vision_layer_idx
        if layer_idx < 0:
            layer_idx = num_vision_layers + layer_idx
        
        # Ensure layer index is valid
        if layer_idx < 0 or layer_idx >= num_vision_layers:
            logger.error(f"Invalid vision layer index: {self.vision_layer_idx}")
            return
        
        # Target the specified layer type
        if self.target_layer == 'final_norm':
            try:
                # LayerNorm after the layer
                vision_module = self.model.vision_encoder.encoder.layer[layer_idx].layernorm_after
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on BLIP vision encoder layer {layer_idx} final norm")
            except (AttributeError, IndexError) as e:
                logger.error(f"Could not find BLIP vision encoder final norm in layer {layer_idx}: {str(e)}")
        
        elif self.target_layer == 'attention':
            try:
                # Self-attention output
                vision_module = self.model.vision_encoder.encoder.layer[layer_idx].attention.output.dense
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on BLIP vision encoder layer {layer_idx} attention output")
            except (AttributeError, IndexError) as e:
                logger.error(f"Could not find BLIP vision encoder attention in layer {layer_idx}: {str(e)}")
        
        elif self.target_layer == 'mlp':
            try:
                # MLP output 
                vision_module = self.model.vision_encoder.encoder.layer[layer_idx].intermediate.output.dense
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on BLIP vision encoder layer {layer_idx} MLP output")
            except (AttributeError, IndexError) as e:
                logger.error(f"Could not find BLIP vision encoder MLP in layer {layer_idx}: {str(e)}")
    
    def _forward_hook(self, module, input_tensor, output_tensor):
        """
        Hook function that applies tensor decomposition to layer outputs
        
        Args:
            module: The layer module
            input_tensor: Input to the layer
            output_tensor: Output from the layer
        
        Returns:
            Modified output with tensor decomposition applied
        """
        # Apply tensor decomposition
        defended_embeddings = self._apply_decomposition(output_tensor)
        
        # Apply residual connection
        final_output = self.alpha * output_tensor + (1 - self.alpha) * defended_embeddings
        
        return final_output
    
    def _apply_decomposition(self, tensor):
        """
        Apply the selected decomposition method to the tensor
        
        Args:
            tensor: Input tensor to decompose
            
        Returns:
            Reconstructed tensor after decomposition
        """
        try:
            if self.method == 'cp':
                # Reshape to 3D tensor for CP decomposition (batch_size, sequence_length, hidden_dim)
                original_shape = tensor.shape
                
                # If tensor is already 3D, use it as is
                if len(original_shape) == 3:
                    tensor_3d = tensor
                else:
                    # If it's 2D, add a dummy dimension
                    tensor_3d = tensor.unsqueeze(1)
                
                # Apply CP decomposition with random initialization
                factors = parafac(tensor_3d, rank=self.rank, init='random', 
                                 tol=1e-4, n_iter_max=50, verbose=False)
                reconstructed = tl.cp_to_tensor(factors)
                
                # Reshape back to original shape if needed
                if len(original_shape) == 2:
                    reconstructed = reconstructed.squeeze(1)
                
            elif self.method == 'tucker':
                # Reshape for Tucker decomposition
                original_shape = tensor.shape
                
                # If tensor is already 3D, use it as is
                if len(original_shape) == 3:
                    tensor_3d = tensor
                    # Define ranks carefully for 3D tensor
                    ranks = [
                        min(original_shape[0], self.rank), 
                        min(original_shape[1], self.rank), 
                        min(original_shape[2], self.rank)
                    ]
                else:
                    # If it's 2D, add a dummy dimension
                    tensor_3d = tensor.unsqueeze(1)
                    # Define ranks for 2D tensor with added dummy dimension
                    ranks = [
                        min(original_shape[0], self.rank), 
                        1, 
                        min(original_shape[1], self.rank)
                    ]
                
                # Apply Tucker decomposition with random initialization
                core, factors = tucker(tensor_3d, rank=ranks, init='random', 
                                      tol=1e-4, n_iter_max=50, verbose=False)
                reconstructed = tl.tucker_to_tensor((core, factors))
                
                # Reshape back to original shape if needed
                if len(original_shape) == 2:
                    reconstructed = reconstructed.squeeze(1)
            
            elif self.method == 'tt':
                # Tensor Train decomposition
                original_shape = tensor.shape
                
                # If tensor is already 3D, use it as is
                if len(original_shape) == 3:
                    tensor_3d = tensor
                else:
                    # If it's 2D, add a dummy dimension
                    tensor_3d = tensor.unsqueeze(1)
                
                # Define ranks for tensor train
                if len(original_shape) == 3:
                    tt_ranks = [1] + [self.rank] * (len(original_shape) - 1) + [1]
                else:
                    tt_ranks = [1, self.rank, 1]
                
                # Apply Tensor Train decomposition
                factors = tensor_train(tensor_3d, tt_ranks, verbose=False)
                reconstructed = tl.tt_to_tensor(factors)
                
                # Reshape back to original shape if needed
                if len(original_shape) == 2:
                    reconstructed = reconstructed.squeeze(1)
            
            else:
                raise ValueError(f"Unknown decomposition method: {self.method}")
                
            return reconstructed
            
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}. Using original tensor.")
            return tensor


class MultiLayerTensorDefense(BaseTensorDefense):
    """Multi-layer tensor decomposition defense
    
    This class applies tensor decomposition to multiple layers 
    simultaneously to provide stronger defense against adversarial attacks.
    """
    
    def __init__(self, model, layer_configs):
        """
        Initialize the multi-layer tensor defense
        
        Args:
            model: CLIP/BLIP model to defend
            layer_configs: List of dictionaries, each containing:
                - 'method': Decomposition method ('cp', 'tucker', 'tt')
                - 'rank': Rank for decomposition
                - 'alpha': Weight for residual connection
                - 'target_layer': Type of layer ('final_norm', 'attention', 'mlp')
                - 'vision_layer_idx': Layer index to apply defense to
        """
        super().__init__(model)
        self.layer_configs = layer_configs
        self.defenses = []
        
        # Create individual defenses for each layer
        for config in layer_configs:
            defense = TargetedTensorDefense(
                model=model,
                method=config.get('method', 'cp'),
                rank=config.get('rank', 64),
                alpha=config.get('alpha', 0.5),
                target_layer=config.get('target_layer', 'final_norm'),
                vision_layer_idx=config.get('vision_layer_idx', -1)
            )
            self.defenses.append(defense)
        
        logger.info(f"Initialized multi-layer defense with {len(layer_configs)} layers")
    
    def remove_hooks(self):
        """Remove hooks from all defense layers"""
        for defense in self.defenses:
            defense.remove_hooks()
        self.defenses = []
        logger.info("All multi-layer defense hooks removed")


class AdaptiveTensorDefense(BaseTensorDefense):
    """Adaptive tensor decomposition defense
    
    This class dynamically selects which layers to defend and what
    decomposition parameters to use based on input characteristics.
    """
    
    def __init__(self, model, defense_configs, detector_model=None):
        """
        Initialize the adaptive tensor defense
        
        Args:
            model: CLIP/BLIP model to defend
            defense_configs: List of different defense configurations to choose from
            detector_model: Optional model that determines which defense to use
        """
        super().__init__(model)
        self.defense_configs = defense_configs
        self.detector_model = detector_model
        self.current_defense = None
        
        logger.info(f"Initialized adaptive defense with {len(defense_configs)} configurations")
    
    def select_defense(self, inputs):
        """
        Select the appropriate defense based on input characteristics
        
        Args:
            inputs: Input tensors to the model
            
        Returns:
            Selected defense configuration index
        """
        if self.detector_model is not None:
            # Use detector model to select defense
            with torch.no_grad():
                detection_score = self.detector_model(inputs)
                # Higher score indicates higher likelihood of attack
                detection_score = torch.sigmoid(detection_score).item()
                
                # Select defense based on detection score
                if detection_score < 0.3:
                    # Low probability of attack - use light defense
                    defense_idx = 0
                elif detection_score < 0.7:
                    # Medium probability - use medium defense
                    defense_idx = 1
                else:
                    # High probability - use strong defense
                    defense_idx = 2
                
                return min(defense_idx, len(self.defense_configs) - 1)
        else:
            # Simple heuristic based on input statistics
            if 'pixel_values' in inputs:
                pixel_values = inputs['pixel_values']
                
                # Calculate gradient magnitude as a simple perturbation indicator
                pixel_values.requires_grad_(True)
                outputs = self.model(**inputs)
                if hasattr(outputs, 'logits_per_image'):
                    loss = outputs.logits_per_image.mean()
                else:
                    loss = outputs.image_embeds.mean()
                
                loss.backward()
                grad_magnitude = torch.abs(pixel_values.grad).mean().item()
                pixel_values.requires_grad_(False)
                
                # Select defense based on gradient magnitude
                if grad_magnitude < 0.01:
                    defense_idx = 0  # Light defense
                elif grad_magnitude < 0.05:
                    defense_idx = 1  # Medium defense
                else:
                    defense_idx = 2  # Strong defense
                
                return min(defense_idx, len(self.defense_configs) - 1)
            
            # Default to middle configuration
            return min(1, len(self.defense_configs) - 1)
    
    def apply_defense(self, inputs):
        """
        Apply the selected defense configuration
        
        Args:
            inputs: Input tensors to the model
            
        Returns:
            Defended model output
        """
        # Remove any existing defense
        if self.current_defense is not None:
            self.current_defense.remove_hooks()
        
        # Select appropriate defense
        defense_idx = self.select_defense(inputs)
        config = self.defense_configs[defense_idx]
        
        # Apply selected defense
        if isinstance(config, list):
            # Multi-layer defense
            self.current_defense = MultiLayerTensorDefense(self.model, config)
        else:
            # Single-layer defense
            self.current_defense = TargetedTensorDefense(
                model=self.model,
                method=config.get('method', 'cp'),
                rank=config.get('rank', 64),
                alpha=config.get('alpha', 0.5),
                target_layer=config.get('target_layer', 'final_norm'),
                vision_layer_idx=config.get('vision_layer_idx', -1)
            )
        
        # Forward pass with defense applied
        outputs = self.model(**inputs)
        
        return outputs
    
    def remove_hooks(self):
        """Remove current defense hooks"""
        if self.current_defense is not None:
            self.current_defense.remove_hooks()
            self.current_defense = None


###############################
# Adversarial Attack Classes  #
###############################

class PGDAttack:
    """
    Projected Gradient Descent (PGD) attack for Vision-Language Models
    
    This class implements a white-box PGD attack that minimizes the similarity 
    between matched image-text pairs in VLM embeddings.
    """
    def __init__(self, model, processor, epsilon=8/255, alpha=6/255, steps=2, 
                 random_start=True, targeted=False):
        """
        Initialize a PGD attack for VLMs
        
        Args:
            model: VLM model to attack
            processor: Model processor for preprocessing
            epsilon: Maximum perturbation magnitude (L-infinity norm)
            alpha: Step size for each iteration
            steps: Number of attack steps
            random_start: Whether to use random initialization
            targeted: Whether to perform a targeted attack
        """
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.targeted = targeted
        
        logging.info(f"Initialized PGD attack with epsilon={epsilon}, alpha={alpha}, steps={steps}")
        logging.info(f"Random start: {random_start}, Targeted: {targeted}")
    
    def perturb(self, images, captions, device, target_captions=None):
        """
        Generate adversarial examples using PGD
        
        Args:
            images: List of PIL images
            captions: List of text captions
            device: Device to run the attack on
            target_captions: Target captions for targeted attack (optional)
            
        Returns:
            Tensor of adversarial examples, inputs tensor, original pixel values
        """
        # Convert PIL images to tensors using the processor
        inputs = self.processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Prepare target captions for targeted attack if specified
        if self.targeted and target_captions is not None:
            target_inputs = self.processor(
                text=target_captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
        
        # Get the pixel values tensor (original images)
        pixel_values = inputs.pixel_values.clone().detach()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize adversarial examples with original pixel values
        adv_images = pixel_values.clone().detach()
        
        # Random initialization for PGD (if enabled)
        if self.random_start and self.steps > 1:
            # Use fixed seed for deterministic results
            torch.manual_seed(42)
            # Add small random noise to start
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon/2, self.epsilon/2)
            adv_images = torch.clamp(adv_images, 0, 1)
        
        # Perform multi-step attack
        for step in range(self.steps):
            # Important: create a fresh copy that requires gradients
            adv_images = adv_images.clone().detach().requires_grad_(True)
            
            # Forward pass - with gradient computation enabled
            with torch.enable_grad():
                # For CLIP-like models with get_image_features method
                if hasattr(self.model, 'get_image_features'):
                    # Get image features
                    image_features = self.model.get_image_features(pixel_values=adv_images)
                    
                    # Get text features
                    text_features = self.model.get_text_features(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
                    
                    if self.targeted and target_captions is not None:
                        target_text_features = self.model.get_text_features(
                            input_ids=target_inputs.input_ids,
                            attention_mask=target_inputs.attention_mask
                        )
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    if self.targeted and target_captions is not None:
                        target_text_features = target_text_features / target_text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarity scores
                    logits_per_image = torch.matmul(image_features, text_features.t())
                    
                    if self.targeted and target_captions is not None:
                        target_logits = torch.matmul(image_features, target_text_features.t())
                    
                    # For the attack, we want to minimize the similarity between matched pairs
                    batch_size = logits_per_image.size(0)
                    targets = torch.arange(batch_size).to(device)
                    
                    # Get the matched pair similarities (diagonal elements)
                    matched_similarities = logits_per_image[torch.arange(batch_size), targets]
                    
                    if self.targeted and target_captions is not None:
                        # For targeted attack, maximize similarity to target captions
                        target_similarities = target_logits[torch.arange(batch_size), targets]
                        # Minimize original similarity, maximize target similarity
                        loss = -target_similarities.sum() + matched_similarities.sum()
                    else:
                        # For untargeted attack, minimize similarity to original captions
                        loss = -matched_similarities.sum()
                    
                else:
                    # For BLIP-like models
                    adv_inputs = inputs.copy()
                    adv_inputs['pixel_values'] = adv_images
                    
                    outputs = self.model(**adv_inputs)
                    
                    if self.targeted and target_captions is not None:
                        target_outputs = self.model(
                            input_ids=target_inputs.input_ids,
                            attention_mask=target_inputs.attention_mask,
                            pixel_values=adv_images
                        )
                    
                    # For untargeted attack, minimize similarity score
                    if hasattr(outputs, 'logits_per_image'):
                        # Get the diagonal elements (matched pairs)
                        batch_size = outputs.logits_per_image.size(0)
                        targets = torch.arange(batch_size).to(device)
                        matched_similarities = outputs.logits_per_image[torch.arange(batch_size), targets]
                        
                        if self.targeted and target_captions is not None:
                            target_similarities = target_outputs.logits_per_image[torch.arange(batch_size), targets]
                            loss = -target_similarities.sum() + matched_similarities.sum()
                        else:
                            loss = -matched_similarities.sum()
                    else:
                        # Fallback for other model types
                        loss = -outputs.similarity_scores.mean()
            
            # Zero all existing gradients
            self.model.zero_grad()
            
            # Compute gradients
            loss.backward()
            
            # Check if gradients were properly computed
            if adv_images.grad is None:
                logging.error("No gradients computed in attack - check autograd setup")
                break
                
            # Update adversarial images
            with torch.no_grad():
                # Take a step in the gradient direction
                adv_images = adv_images.detach() + self.alpha * adv_images.grad.sign()
                
                # Project back to epsilon ball around original images
                delta = torch.clamp(adv_images - pixel_values, min=-self.epsilon, max=self.epsilon)
                
                # Ensure we stay in valid image range [0,1]
                adv_images = torch.clamp(pixel_values + delta, min=0, max=1).detach()
        
        return adv_images, inputs, pixel_values


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) attack for Vision-Language Models
    
    A simple one-step white-box attack that adds gradient-based perturbation
    to maximize the loss.
    """
    def __init__(self, model, processor, epsilon=8/255):
        """
        Initialize an FGSM attack for VLMs
        
        Args:
            model: VLM model to attack
            processor: Model processor for preprocessing
            epsilon: Perturbation magnitude
        """
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        
        logging.info(f"Initialized FGSM attack with epsilon={epsilon}")
    
    def perturb(self, images, captions, device):
        """
        Generate adversarial examples using FGSM
        
        Args:
            images: List of PIL images
            captions: List of text captions
            device: Device to run the attack on
            
        Returns:
            Tensor of adversarial examples, inputs tensor, original pixel values
        """
        # Same implementation as PGD but with a single step
        pgd_attack = PGDAttack(
            model=self.model,
            processor=self.processor,
            epsilon=self.epsilon,
            alpha=self.epsilon,
            steps=1,
            random_start=False
        )
        
        return pgd_attack.perturb(images, captions, device)


###############################
# Dataset and Data Utilities  #
###############################

class HFDatasetWrapper(Dataset):
    """
    Dataset wrapper for HuggingFace dataset
    """
    def __init__(self, hf_dataset, split="test", max_samples=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            hf_dataset: HuggingFace dataset
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to use (None = use all)
            transform: Optional transform to apply to images
        """
        self.hf_dataset = hf_dataset[split]
        self.transform = transform
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples > 0:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
            
        logger.info(f"Initialized dataset with {len(self.hf_dataset)} samples")
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Get image - should be a PIL Image or convertible to one
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        elif 'pixel_values' in item:
            # Convert tensor to PIL image if needed
            if isinstance(item['pixel_values'], torch.Tensor):
                image = ToPILImage()(item['pixel_values'].squeeze())
            else:
                image = item['pixel_values']
        else:
            raise ValueError(f"No image field found in dataset item: {item.keys()}")
        
        # Apply transform if specified
        if self.transform is not None:
            image = self.transform(image)
        
        # Get caption (use first caption if there are multiple)
        if 'caption' in item:
            captions = item['caption']
            caption = captions[0] if isinstance(captions, list) else captions
        elif 'text' in item:
            caption = item['text']
        else:
            caption = ""  # Empty caption as fallback
        
        # Get image id 
        if 'filename' in item:
            image_id = item['filename']
        elif 'img_id' in item:
            image_id = str(item['img_id'])
        elif 'image_id' in item:
            image_id = str(item['image_id'])
        else:
            image_id = str(idx)
        
        return {
            'image_id': image_id,
            'image': image,
            'caption': caption
        }


# Custom collate function to handle PIL images
def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images
    
    Args:
        batch: Batch of samples
        
    Returns:
        Dictionary of batched data
    """
    image_ids = [item['image_id'] for item in batch]
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    return {
        'image_id': image_ids,
        'image': images,
        'caption': captions
    }


###############################
# Evaluation Utilities        #
###############################

def calculate_recall_at_k(similarity, targets, k=1):
    """
    Calculate recall@k from similarity matrix
    
    Args:
        similarity: Similarity matrix of shape [batch_size, batch_size]
        targets: Ground truth indices
        k: k value for recall@k
    
    Returns:
        Recall@k value
    """
    # Ensure k is at least 1 and at most the width of the similarity matrix
    k = max(1, min(k, similarity.shape[1]))
    
    # Get top-k indices
    _, indices = similarity.topk(k, dim=1)
    correct = torch.any(indices == targets.view(-1, 1), dim=1).float()
    recall = correct.mean().item()
    return recall


def calculate_metrics(clean_similarity, adv_no_defense, adv_with_defense):
    """
    Calculate comprehensive metrics comparing clean, attacked, and defended performance
    
    Args:
        clean_similarity: Similarity matrix for clean images
        adv_no_defense: Similarity matrix for adversarial images without defense
        adv_with_defense: Similarity matrix for adversarial images with defense
        
    Returns:
        Dictionary of metrics
    """
    batch_size = clean_similarity.size(0)
    targets = torch.arange(batch_size).to(clean_similarity.device)
    
    metrics = {}
    
    # Calculate recall@k for k in [1, 5, 10]
    for k in [1, 5, 10]:
        if batch_size >= k:
            # Clean performance
            clean_recall = calculate_recall_at_k(clean_similarity, targets, k)
            metrics[f'clean_recall_at_{k}'] = clean_recall
            
            # Attacked performance (no defense)
            no_defense_recall = calculate_recall_at_k(adv_no_defense, targets, k)
            metrics[f'no_defense_recall_at_{k}'] = no_defense_recall
            
            # Defended performance
            defended_recall = calculate_recall_at_k(adv_with_defense, targets, k)
            metrics[f'defended_recall_at_{k}'] = defended_recall
            
            # Calculate improvements
            abs_improvement = defended_recall - no_defense_recall
            metrics[f'abs_improvement_at_{k}'] = abs_improvement
            
            rel_improvement = (abs_improvement / no_defense_recall * 100) if no_defense_recall > 0 else 0
            metrics[f'rel_improvement_at_{k}'] = rel_improvement
            
            # Calculate recovery percentage
            attack_drop = clean_recall - no_defense_recall
            defense_drop = clean_recall - defended_recall
            
            if attack_drop > 0:
                recovery = (attack_drop - defense_drop) / attack_drop * 100
                metrics[f'recovery_percent_at_{k}'] = recovery
            else:
                metrics[f'recovery_percent_at_{k}'] = 0
    
    return metrics


def generate_adversarial_examples(model, processor, dataloader, attack_config, device):
    """
    Generate adversarial examples for dataset evaluation
    
    Args:
        model: Model to attack
        processor: Image/text processor
        dataloader: DataLoader for evaluation data
        attack_config: Dictionary with attack parameters
        device: Device to run attack on
        
    Returns:
        Dictionary of adversarial examples and corresponding inputs
    """
    attack_type = attack_config.get('type', 'pgd')
    epsilon = attack_config.get('epsilon', 8/255)
    steps = attack_config.get('steps', 2)
    step_size = attack_config.get('step_size', 6/255)
    
    logger.info(f"Generating adversarial examples using {attack_type.upper()} attack")
    logger.info(f"Attack parameters: epsilon={epsilon}, steps={steps}, step_size={step_size}")
    
    # Initialize attack
    if attack_type.lower() == 'pgd':
        attack = PGDAttack(
            model, processor, 
            epsilon=epsilon, 
            alpha=step_size, 
            steps=steps
        )
    elif attack_type.lower() == 'fgsm':
        attack = FGSM(
            model, processor,
            epsilon=epsilon
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    # Storage for adversarial examples
    adv_examples = {}
    
    # Track the first batch for sample images
    sample_originals = None
    sample_adversarials = None
    sample_captions = None
    
    # Batch processing with progress bar
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating adversarial examples")):
        try:
            image_ids = batch['image_id']
            images = batch['image']
            captions = batch['caption']
            
            # Generate adversarial examples
            perturbed_pixel_values, inputs, original_pixels = attack.perturb(images, captions, device)
            
            # Store the first batch for sample images
            if batch_idx == 0:
                sample_originals = original_pixels
                sample_adversarials = perturbed_pixel_values
                sample_captions = captions
            
            # Process clean images to get text embeddings (needed later)
            with torch.no_grad():
                # For CLIP-like models
                if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
                    clean_image_embeds = model.get_image_features(pixel_values=original_pixels)
                    text_embeds = model.get_text_features(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
                    
                    # Normalize embeddings
                    clean_image_embeds = clean_image_embeds / clean_image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate clean similarity scores
                    clean_similarity = torch.matmul(clean_image_embeds, text_embeds.t())
                
                # For BLIP-like models
                else:
                    clean_outputs = model(**inputs)
                    if hasattr(clean_outputs, 'logits_per_image'):
                        clean_similarity = clean_outputs.logits_per_image
                    else:
                        clean_similarity = clean_outputs.similarity_scores
                    
                    # Store text embeddings if available
                    if hasattr(clean_outputs, 'text_embeds'):
                        text_embeds = clean_outputs.text_embeds
                    else:
                        text_embeds = None
                
                # Store in dictionary
                adv_examples[batch_idx] = {
                    'image_ids': image_ids,
                    'perturbed_pixel_values': perturbed_pixel_values.cpu(),  # Move to CPU to save GPU memory
                    'inputs': {
                        'input_ids': inputs.input_ids.cpu(),
                        'attention_mask': inputs.attention_mask.cpu()
                    },
                    'text_embeds': text_embeds.cpu() if text_embeds is not None else None,
                    'clean_similarity': clean_similarity.cpu()
                }
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
            
        # Free up memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logging.info(f"Generated adversarial examples for {len(adv_examples)} batches")
    
    # Save sample images for visualization
    if sample_originals is not None and sample_adversarials is not None:
        save_sample_images(sample_originals, sample_adversarials, sample_captions, 
                          epsilon=epsilon, steps=steps, step_size=step_size)
    
    return adv_examples


def save_sample_images(original_images, adversarial_images, captions, 
                      indices=None, max_samples=5, epsilon=8/255, steps=2, step_size=6/255):
    """
    Save original and adversarial image pairs for visualization
    
    Args:
        original_images: Tensor of original image pixel values
        adversarial_images: Tensor of adversarial image pixel values
        captions: List of corresponding image captions
        indices: Specific indices to save (default: first max_samples)
        max_samples: Maximum number of samples to save
        epsilon: Perturbation magnitude used in attack
        steps: Number of attack steps
        step_size: Step size used in attack
    """
    # Create directory for samples if it doesn't exist
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)
    
    # If no indices provided, use first max_samples
    if indices is None:
        indices = list(range(min(max_samples, len(original_images))))
    else:
        # Limit to max_samples
        indices = indices[:max_samples]
    
    # Save each pair of original and adversarial images
    for i, idx in enumerate(indices):
        try:
            # Get original and adversarial images
            orig_img = original_images[idx].permute(1, 2, 0).cpu().numpy()
            adv_img = adversarial_images[idx].permute(1, 2, 0).cpu().numpy()
            
            # Fix image data range issues - clip to [0,1] before visualization
            orig_img = np.clip(orig_img, 0, 1)
            adv_img = np.clip(adv_img, 0, 1)
            
            # Get caption
            caption = captions[idx]
            if len(caption) > 50:
                caption = caption[:47] + "..."
            
            # Create figure with subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot original image
            axs[0].imshow(orig_img)
            axs[0].set_title("Original Image", fontsize=14)
            axs[0].axis('off')
            
            # Plot adversarial image
            axs[1].imshow(adv_img)
            axs[1].set_title(f"Adversarial Image\nε={epsilon:.4f}, steps={steps}", fontsize=14)
            axs[1].axis('off')
            
            # Plot perturbation (difference)
            perturbation = np.abs(adv_img - orig_img)
            # Normalize for better visualization
            perturbation = perturbation / perturbation.max() if perturbation.max() > 0 else perturbation
            
            axs[2].imshow(perturbation, cmap='viridis')
            axs[2].set_title("Perturbation Visualization\n(Scaled for visibility)", fontsize=14)
            axs[2].axis('off')
            
            # Add caption as suptitle
            plt.suptitle(f"Caption: {caption}", fontsize=16)
            
            # Add attack parameters text
            attack_params = (
                f"Attack Parameters:\n"
                f"Epsilon: {epsilon:.4f}, Steps: {steps}, Step Size: {step_size:.4f}"
            )
            fig.text(0.5, 0.01, attack_params, ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.1)
            
            # Save figure
            plt.savefig(os.path.join(SAMPLES_DIR, f"sample_{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(SAMPLES_DIR, f"sample_{i+1}.pdf"), bbox_inches='tight')
            plt.close()
            
            # Also save as individual PNGs for inclusion in reports
            # Convert tensors to PIL images - ensure proper range [0,255]
            orig_img_pil = Image.fromarray((orig_img * 255).astype(np.uint8))
            adv_img_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
            
            # Save individual images
            orig_img_pil.save(os.path.join(SAMPLES_DIR, f"original_{i+1}.png"))
            adv_img_pil.save(os.path.join(SAMPLES_DIR, f"adversarial_{i+1}.png"))
            
            print(f"Saved sample pair {i+1}/{len(indices)}")
            
        except Exception as e:
            logging.error(f"Error saving sample {i+1}: {str(e)}")
    
    # Save attack parameters in a text file
    with open(os.path.join(SAMPLES_DIR, "attack_parameters.txt"), 'w') as f:
        f.write(f"Attack Parameters:\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Step Size: {step_size}\n")
    
    print(f"Saved {len(indices)} sample image pairs to {SAMPLES_DIR}")


def evaluate_defense(model, processor, dataloader, adv_examples, defense=None, 
                    defense_name="No Defense", device=None):
    """
    Evaluate model with tensor decomposition defense
    
    Args:
        model: Model to evaluate
        processor: Image/text processor
        dataloader: DataLoader for evaluation data
        adv_examples: Pre-generated adversarial examples
        defense: Defense configuration or None
        defense_name: Name of the defense for printing
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For storing results
    all_recalls_clean = {1: [], 5: [], 10: []}
    all_recalls_adv = {1: [], 5: [], 10: []}
    all_recalls_defended = {1: [], 5: [], 10: []}
    
    # For tracking metrics per batch
    batch_metrics = []
    
    batch_count = 0
    
    # Metrics to track
    metrics = {
        'image_ids': [],
        'defense_name': defense_name,
    }
    
    # Empty CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Apply defense if specified
    defense_instance = None
    if defense is not None:
        if isinstance(defense, dict):
            # Single layer defense
            defense_instance = TargetedTensorDefense(
                model=model,
                method=defense.get('method', 'cp'),
                rank=defense.get('rank', 64),
                alpha=defense.get('alpha', 0.5),
                target_layer=defense.get('target_layer', 'final_norm'),
                vision_layer_idx=defense.get('vision_layer_idx', -1)
            )
        elif isinstance(defense, list):
            # Multi-layer defense
            defense_instance = MultiLayerTensorDefense(model, defense)
        elif isinstance(defense, BaseTensorDefense):
            # Already instantiated defense
            defense_instance = defense
    
    # Batch processing with progress bar
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {defense_name}")):
        # Skip if this batch wasn't in adv_examples
        if batch_idx not in adv_examples:
            logger.warning(f"Batch {batch_idx} not found in adversarial examples, skipping")
            continue
        
        # Get pre-generated adversarial examples
        adv_batch = adv_examples[batch_idx]
        image_ids = adv_batch['image_ids']
        perturbed_pixel_values = adv_batch['perturbed_pixel_values'].to(device)
        clean_similarity = adv_batch['clean_similarity'].to(device)
        
        # Get input tensors
        input_ids = adv_batch['inputs']['input_ids'].to(device)
        attention_mask = adv_batch['inputs']['attention_mask'].to(device)
        
        text_embeds = None
        if adv_batch['text_embeds'] is not None:
            text_embeds = adv_batch['text_embeds'].to(device)
        
        # First evaluate without defense - temporarily remove hooks if they exist
        if defense_instance is not None:
            defense_instance.remove_hooks()
        
        with torch.no_grad():
            # Prepare inputs for the model
            adv_inputs = {
                'pixel_values': perturbed_pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Get no-defense outputs
            if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
                # CLIP-like model
                adv_image_embeds = model.get_image_features(pixel_values=perturbed_pixel_values)
                
                if text_embeds is None:
                    text_embeds = model.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Normalize embeddings
                adv_image_embeds = adv_image_embeds / adv_image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate adversarial similarity scores
                adv_similarity = torch.matmul(adv_image_embeds, text_embeds.t())
            else:
                # BLIP or other model
                adv_outputs = model(**adv_inputs)
                
                if hasattr(adv_outputs, 'logits_per_image'):
                    adv_similarity = adv_outputs.logits_per_image
                else:
                    adv_similarity = adv_outputs.similarity_scores
        
        # Apply defense and evaluate with defense
        if defense_instance is not None:
            # Reapply defense
            if isinstance(defense, dict):
                defense_instance = TargetedTensorDefense(
                    model=model,
                    method=defense.get('method', 'cp'),
                    rank=defense.get('rank', 64),
                    alpha=defense.get('alpha', 0.5),
                    target_layer=defense.get('target_layer', 'final_norm'),
                    vision_layer_idx=defense.get('vision_layer_idx', -1)
                )
            elif isinstance(defense, list):
                defense_instance = MultiLayerTensorDefense(model, defense)
            
            # Process adversarial images with defense
            with torch.no_grad():
                if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
                    # CLIP-like model
                    defended_image_embeds = model.get_image_features(pixel_values=perturbed_pixel_values)
                    
                    # Normalize embeddings
                    defended_image_embeds = defended_image_embeds / defended_image_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate defended similarity scores
                    defended_similarity = torch.matmul(defended_image_embeds, text_embeds.t())
                else:
                    # BLIP or other model
                    defended_outputs = model(**adv_inputs)
                    
                    if hasattr(defended_outputs, 'logits_per_image'):
                        defended_similarity = defended_outputs.logits_per_image
                    else:
                        defended_similarity = defended_outputs.similarity_scores
        else:
            # No defense applied
            defended_similarity = adv_similarity
        
        # Store image IDs
        metrics['image_ids'].extend(image_ids)
        
        # Calculate retrieval metrics for this batch
        batch_size = len(image_ids)
        targets = torch.arange(batch_size).to(device)
        
        # Calculate recall@k for clean, adversarial, and defended
        batch_result = {'batch': batch_idx, 'image_ids': image_ids}
        
        for k in [1, 5, 10]:
            if batch_size >= k:
                clean_recall = calculate_recall_at_k(clean_similarity, targets, k)
                adv_recall = calculate_recall_at_k(adv_similarity, targets, k)
                defended_recall = calculate_recall_at_k(defended_similarity, targets, k)
                
                all_recalls_clean[k].append(clean_recall)
                all_recalls_adv[k].append(adv_recall)
                all_recalls_defended[k].append(defended_recall)
                
                # Store batch metrics
                batch_result[f'clean_recall_at_{k}'] = clean_recall
                batch_result[f'adv_recall_at_{k}'] = adv_recall
                batch_result[f'defended_recall_at_{k}'] = defended_recall
                
                # Calculate improvements
                improvement = defended_recall - adv_recall
                batch_result[f'improvement_at_{k}'] = improvement
                
                # Calculate recovery
                if clean_recall > adv_recall:
                    recovery = (defended_recall - adv_recall) / (clean_recall - adv_recall) * 100
                    batch_result[f'recovery_percent_at_{k}'] = recovery
                else:
                    batch_result[f'recovery_percent_at_{k}'] = 0
        
        batch_metrics.append(batch_result)
        batch_count += 1
        
        # Free up memory after each batch
        if torch.cuda.is_available():
            del perturbed_pixel_values, text_embeds, clean_similarity
            del input_ids, attention_mask
            torch.cuda.empty_cache()
    
    # Remove the hooks after evaluation
    if defense_instance is not None:
        defense_instance.remove_hooks()
    
    # Calculate average recall across all batches
    for k in [1, 5, 10]:
        if all_recalls_clean[k]:
            metrics[f'clean_recall_at_{k}'] = np.mean(all_recalls_clean[k])
            metrics[f'adv_recall_at_{k}'] = np.mean(all_recalls_adv[k])
            metrics[f'defended_recall_at_{k}'] = np.mean(all_recalls_defended[k])
    
    # Calculate performance improvements and recovery
    for k in [1, 5, 10]:
        clean_recall = metrics[f'clean_recall_at_{k}']
        adv_recall = metrics[f'adv_recall_at_{k}']
        defended_recall = metrics[f'defended_recall_at_{k}']
        
        # Improvement over no defense
        metrics[f'improvement_at_{k}'] = defended_recall - adv_recall
        if adv_recall > 0:
            metrics[f'improvement_percent_at_{k}'] = (defended_recall - adv_recall) / adv_recall * 100
        else:
            metrics[f'improvement_percent_at_{k}'] = 0
        
        # Drop from clean performance
        metrics[f'clean_to_adv_drop_at_{k}'] = clean_recall - adv_recall
        metrics[f'clean_to_defended_drop_at_{k}'] = clean_recall - defended_recall
        
        # Recovery percentage
        attack_drop = clean_recall - adv_recall
        if attack_drop > 0:
            metrics[f'recovery_percent_at_{k}'] = (defended_recall - adv_recall) / attack_drop * 100
        else:
            metrics[f'recovery_percent_at_{k}'] = 0
    
    # Print results
    logger.info(f"\n{defense_name} Evaluation Results (averaged over {batch_count} batches):")
    for k in [1, 5, 10]:
        clean_recall = metrics[f'clean_recall_at_{k}']
        adv_recall = metrics[f'adv_recall_at_{k}']
        defended_recall = metrics[f'defended_recall_at_{k}']
        improvement_percent = metrics[f'improvement_percent_at_{k}']
        recovery_percent = metrics[f'recovery_percent_at_{k}']
        
        logger.info(f"  Recall@{k}:")
        logger.info(f"    Clean: {clean_recall:.4f}")
        logger.info(f"    Adversarial (No Defense): {adv_recall:.4f}")
        logger.info(f"    Adversarial (With {defense_name}): {defended_recall:.4f}")
        logger.info(f"    Improvement: {improvement_percent:.2f}%")
        logger.info(f"    Performance Recovery: {recovery_percent:.2f}% of the adversarial drop")
    
    # Save batch metrics to CSV
    defense_file_name = defense_name.lower().replace(' ', '_').replace('(','').replace(')','')
    pd.DataFrame(batch_metrics).to_csv(os.path.join(RESULTS_DIR, f"{defense_file_name}_batch_metrics.csv"), index=False)
    
    return metrics, batch_metrics


###############################
# Visualization Functions     #
###############################

def create_defense_comparison_plot(all_results, k=1, save_path=None):
    """
    Create bar chart comparing defense techniques
    
    Args:
        all_results: List of result dictionaries from evaluate_defense
        k: k value for Recall@k to compare
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, f"defense_comparison_r{k}.pdf")
    
    # Extract data from results
    defense_names = [result['defense_name'] for result in all_results]
    clean_recalls = [result[f'clean_recall_at_{k}'] * 100 for result in all_results]
    adv_recalls = [result[f'adv_recall_at_{k}'] * 100 for result in all_results]
    defended_recalls = [result[f'defended_recall_at_{k}'] * 100 for result in all_results]
    
    # Sort by defended recall
    sorted_indices = np.argsort(defended_recalls)[::-1]  # Descending order
    defense_names = [defense_names[i] for i in sorted_indices]
    clean_recalls = [clean_recalls[i] for i in sorted_indices]
    adv_recalls = [adv_recalls[i] for i in sorted_indices]
    defended_recalls = [defended_recalls[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(defense_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, clean_recalls, width=barWidth, label='Clean', color='#2ca02c')
    plt.bar(r2, adv_recalls, width=barWidth, label='Adversarial', color='#d62728')
    plt.bar(r3, defended_recalls, width=barWidth, label='Defended', color='#1f77b4')
    
    # Add labels
    plt.xlabel('Defense Method', fontsize=14)
    plt.ylabel(f'Recall@{k} (%)', fontsize=14)
    plt.title(f'Comparison of Defense Methods (Recall@{k})', fontsize=16)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(defense_names))], defense_names, rotation=45, ha='right')
    
    # Create legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    bars1 = plt.bar(r1, clean_recalls, width=barWidth)
    bars2 = plt.bar(r2, adv_recalls, width=barWidth)
    bars3 = plt.bar(r3, defended_recalls, width=barWidth)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_recovery_plot(all_results, save_path=None):
    """
    Create plot showing recovery percentage for different defenses
    
    Args:
        all_results: List of result dictionaries from evaluate_defense
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "recovery_comparison.pdf")
    
    # Extract data from results
    defense_names = [result['defense_name'] for result in all_results]
    recovery_r1 = [result.get('recovery_percent_at_1', 0) for result in all_results]
    recovery_r5 = [result.get('recovery_percent_at_5', 0) for result in all_results]
    recovery_r10 = [result.get('recovery_percent_at_10', 0) for result in all_results]
    
    # Create a dataframe for easier plotting
    data = {
        'Defense': defense_names,
        'Recall@1': recovery_r1,
        'Recall@5': recovery_r5,
        'Recall@10': recovery_r10
    }
    df = pd.DataFrame(data)
    
    # Sort by Recall@1 recovery
    df = df.sort_values('Recall@1', ascending=False)
    
    # Reshape for plotting
    df_melted = df.melt(id_vars='Defense', var_name='Metric', value_name='Recovery (%)')
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    ax = sns.barplot(x='Defense', y='Recovery (%)', hue='Metric', data=df_melted)
    
    # Customize
    plt.title('Recovery Percentage by Defense Method', fontsize=16)
    plt.xlabel('Defense Method', fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='', fontsize=12)
    
    # Add a horizontal line at 0%
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_layer_heatmap(layer_results, save_path=None):
    """
    Create heatmap showing performance of defense across different layers
    
    Args:
        layer_results: Dictionary with layer indices as keys and result dictionaries as values
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "layer_heatmap.pdf")
    
    # Extract layer indices and metrics
    layer_indices = sorted(layer_results.keys())
    metrics = ["Recall@1", "Recall@5", "Recovery@1", "Recovery@5"]
    
    # Create data array
    data = np.zeros((len(metrics), len(layer_indices)))
    
    for i, metric in enumerate(metrics):
        for j, layer_idx in enumerate(layer_indices):
            result = layer_results[layer_idx]
            if metric == "Recall@1":
                data[i, j] = result.get('defended_recall_at_1', 0) * 100
            elif metric == "Recall@5":
                data[i, j] = result.get('defended_recall_at_5', 0) * 100
            elif metric == "Recovery@1":
                data[i, j] = result.get('recovery_percent_at_1', 0)
            elif metric == "Recovery@5":
                data[i, j] = result.get('recovery_percent_at_5', 0)
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Create heatmap
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=layer_indices, yticklabels=metrics)
    
    # Customize
    plt.title('Defense Performance Across Layers', fontsize=16)
    plt.xlabel('Layer Index', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_parameter_sensitivity_plot(param_results, param_name, metric='recovery_percent_at_1', save_path=None):
    """
    Create plot showing sensitivity to a specific parameter
    
    Args:
        param_results: Dictionary with parameter values as keys and result dictionaries as values
        param_name: Name of the parameter (e.g., 'rank', 'alpha')
        metric: Metric to plot
        save_path: Path to save the figure
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, f"{param_name}_sensitivity.pdf")
    
    # Extract parameter values and metrics
    param_values = sorted(param_results.keys())
    metric_values = []
    
    for param_val in param_values:
        result = param_results[param_val]
        metric_values.append(result.get(metric, 0))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create line plot with markers
    plt.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
    
    # Add data labels
    for x, y in zip(param_values, metric_values):
        if 'recall' in metric:
            label = f"{y*100:.1f}%"
        else:
            label = f"{y:.1f}%"
        plt.annotate(label, (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    # Customize
    title_metric = metric.replace('_', ' ').replace('at', '@').title()
    plt.title(f'Sensitivity to {param_name.title()} ({title_metric})', fontsize=16)
    plt.xlabel(param_name.title(), fontsize=14)
    
    if 'recall' in metric:
        plt.ylabel('Recall (%)', fontsize=14)
        plt.ylim(0, 100)
    else:
        plt.ylabel('Recovery (%)', fontsize=14)
    
    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_results_table(all_results, save_path=None):
    """
    Create a table summarizing results across all defenses
    
    Args:
        all_results: List of result dictionaries from evaluate_defense
        save_path: Path to save the table as CSV
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "defense_results_summary.csv")
    
    # Create rows for the table
    rows = []
    
    for result in all_results:
        defense_name = result['defense_name']
        
        # Extract key metrics
        clean_r1 = result.get('clean_recall_at_1', 0) * 100
        adv_r1 = result.get('adv_recall_at_1', 0) * 100
        defended_r1 = result.get('defended_recall_at_1', 0) * 100
        improvement_r1 = result.get('improvement_at_1', 0) * 100
        recovery_r1 = result.get('recovery_percent_at_1', 0)
        
        clean_r5 = result.get('clean_recall_at_5', 0) * 100
        adv_r5 = result.get('adv_recall_at_5', 0) * 100
        defended_r5 = result.get('defended_recall_at_5', 0) * 100
        improvement_r5 = result.get('improvement_at_5', 0) * 100
        recovery_r5 = result.get('recovery_percent_at_5', 0)
        
        # Get defense parameters if available
        if isinstance(result.get('defense_config', None), dict):
            config = result['defense_config']
            method = config.get('method', 'N/A')
            rank = config.get('rank', 'N/A')
            alpha = config.get('alpha', 'N/A')
            target_layer = config.get('target_layer', 'N/A')
            layer_idx = config.get('vision_layer_idx', 'N/A')
        else:
            method = 'N/A'
            rank = 'N/A'
            alpha = 'N/A'
            target_layer = 'N/A'
            layer_idx = 'N/A'
        
        # Create row
        row = {
            'Defense': defense_name,
            'Method': method,
            'Rank': rank,
            'Alpha': alpha,
            'Target_Layer': target_layer,
            'Layer_Index': layer_idx,
            'Clean_R@1': f"{clean_r1:.2f}",
            'Adv_R@1': f"{adv_r1:.2f}",
            'Defended_R@1': f"{defended_r1:.2f}",
            'Improvement_R@1': f"{improvement_r1:.2f}",
            'Recovery_R@1': f"{recovery_r1:.2f}",
            'Clean_R@5': f"{clean_r5:.2f}",
            'Adv_R@5': f"{adv_r5:.2f}",
            'Defended_R@5': f"{defended_r5:.2f}",
            'Improvement_R@5': f"{improvement_r5:.2f}",
            'Recovery_R@5': f"{recovery_r5:.2f}"
        }
        
        rows.append(row)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    
    # Also create a formatted version for the report
    styled_df = df.style.format({
        col: "{:.2f}%" for col in df.columns if col.startswith(('Clean_', 'Adv_', 'Defended_', 'Improvement_'))
    }).format({
        col: "{:.2f}%" for col in df.columns if col.startswith('Recovery_')
    })
    
    # Save as HTML for inclusion in reports
    styled_df.to_html(save_path.replace('.csv', '.html'))
    
    return df


###############################
# Experiment Configurations   #
###############################

def get_defense_configurations(config_type='standard'):
    """
    Get defense configurations for experiments
    
    Args:
        config_type: Type of configurations to return
            - 'standard': Basic set of configurations
            - 'comprehensive': Larger set with more variations
            - 'multilayer': Configurations for multi-layer defense
            - 'all': All available configurations
    
    Returns:
        List of defense configurations
    """
    configurations = []
    
    if config_type in ['standard', 'all']:
        # Base configuration - CP decomposition on final norm
        configurations.append({
            'name': 'CP Final Norm (R=64, α=0.5)',
            'config': {
                'method': 'cp',
                'rank': 64,
                'alpha': 0.5,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
        
        # Compare different layer types
        configurations.append({
            'name': 'CP Attention (R=64, α=0.5)',
            'config': {
                'method': 'cp',
                'rank': 64,
                'alpha': 0.5,
                'target_layer': 'attention',
                'vision_layer_idx': -1
            }
        })
        configurations.append({
            'name': 'CP MLP (R=64, α=0.5)',
            'config': {
                'method': 'cp',
                'rank': 64,
                'alpha': 0.5,
                'target_layer': 'mlp',
                'vision_layer_idx': -1
            }
        })
        
        # Compare different alpha values
        configurations.append({
            'name': 'CP Final Norm (R=64, α=0.3)',
            'config': {
                'method': 'cp',
                'rank': 64,
                'alpha': 0.3,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
        configurations.append({
            'name': 'CP Final Norm (R=64, α=0.7)',
            'config': {
                'method': 'cp',
                'rank': 64,
                'alpha': 0.7,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
        
        # Compare different ranks
        configurations.append({
            'name': 'CP Final Norm (R=32, α=0.5)',
            'config': {
                'method': 'cp',
                'rank': 32,
                'alpha': 0.5,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
        configurations.append({
            'name': 'CP Final Norm (R=128, α=0.5)',
            'config': {
                'method': 'cp',
                'rank': 128,
                'alpha': 0.5,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
        
        # Compare different decomposition methods
        configurations.append({
            'name': 'Tucker Final Norm (R=64, α=0.5)',
            'config': {
                'method': 'tucker',
                'rank': 64,
                'alpha': 0.5,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
        configurations.append({
            'name': 'TT Final Norm (R=64, α=0.5)',
            'config': {
                'method': 'tt',
                'rank': 64,
                'alpha': 0.5,
                'target_layer': 'final_norm',
                'vision_layer_idx': -1
            }
        })
    
    if config_type in ['comprehensive', 'all']:
        # Different layers in the transformer
        for layer_idx in [-1, -2, -3, -4, -5]:
            configurations.append({
                'name': f'CP Final Norm (Layer {layer_idx}, R=64, α=0.5)',
                'config': {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': layer_idx
                }
            })
        
        # More extensive parameter sweep for alpha
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            configurations.append({
                'name': f'CP Final Norm (R=64, α={alpha})',
                'config': {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': alpha,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -1
                }
            })
        
        # More extensive parameter sweep for rank
        for rank in [16, 32, 48, 64, 96, 128, 192, 256]:
            configurations.append({
                'name': f'CP Final Norm (R={rank}, α=0.5)',
                'config': {
                    'method': 'cp',
                    'rank': rank,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -1
                }
            })
    
    if config_type in ['multilayer', 'all']:
        # Multi-layer configurations
        
        # Last 2 layers with same configuration
        configurations.append({
            'name': 'CP Multi-Layer (2 Layers)',
            'config': [
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -1
                },
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -2
                }
            ]
        })
        
        # Last 3 layers with same configuration
        configurations.append({
            'name': 'CP Multi-Layer (3 Layers)',
            'config': [
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -1
                },
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -2
                },
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -3
                }
            ]
        })
        
        # Different combinations of 3 consecutive layers
        for start_idx in [-5, -4, -3]:
            configurations.append({
                'name': f'CP Multi-Layer ({start_idx}:{start_idx+2})',
                'config': [
                    {
                        'method': 'cp',
                        'rank': 64,
                        'alpha': 0.5,
                        'target_layer': 'final_norm',
                        'vision_layer_idx': start_idx
                    },
                    {
                        'method': 'cp',
                        'rank': 64,
                        'alpha': 0.5,
                        'target_layer': 'final_norm',
                        'vision_layer_idx': start_idx + 1
                    },
                    {
                        'method': 'cp',
                        'rank': 64,
                        'alpha': 0.5,
                        'target_layer': 'final_norm',
                        'vision_layer_idx': start_idx + 2
                    }
                ]
            })
        
        # Mixed layer types (attention, mlp, norm)
        configurations.append({
            'name': 'Mixed Layer Types',
            'config': [
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -1
                },
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'attention',
                    'vision_layer_idx': -2
                },
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'mlp',
                    'vision_layer_idx': -3
                }
            ]
        })
        
        # Mixed decomposition methods
        configurations.append({
            'name': 'Mixed Decomposition Methods',
            'config': [
                {
                    'method': 'cp',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -1
                },
                {
                    'method': 'tucker',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -2
                },
                {
                    'method': 'tt',
                    'rank': 64,
                    'alpha': 0.5,
                    'target_layer': 'final_norm',
                    'vision_layer_idx': -3
                }
            ]
        })
    
    return configurations


def get_attack_configurations():
    """
    Get attack configurations for experiments
    
    Returns:
        List of attack configurations
    """
    configurations = []
    
    # Standard PGD attack
    configurations.append({
        'name': 'PGD (ε=8/255, steps=2)',
        'type': 'pgd',
        'epsilon': 8/255,
        'steps': 2,
        'step_size': 6/255
    })
    
    # Weaker PGD attack
    configurations.append({
        'name': 'PGD-Weak (ε=4/255, steps=2)',
        'type': 'pgd',
        'epsilon': 4/255,
        'steps': 2,
        'step_size': 3/255
    })
    
    # Stronger PGD attack
    configurations.append({
        'name': 'PGD-Strong (ε=16/255, steps=10)',
        'type': 'pgd',
        'epsilon': 16/255,
        'steps': 10,
        'step_size': 3/255
    })
    
    # FGSM attack
    configurations.append({
        'name': 'FGSM (ε=8/255)',
        'type': 'fgsm',
        'epsilon': 8/255
    })
    
    return configurations


def get_dataset_configurations():
    """
    Get dataset configurations for experiments
    
    Returns:
        List of dataset configurations
    """
    configurations = []
    
    # Flickr30k (smaller size for quick testing)
    configurations.append({
        'name': 'Flickr30k-Small',
        'dataset': 'nlphuji/flickr30k',
        'split': 'test',
        'max_samples': 1000
    })
    
    # Flickr30k (full test set)
    configurations.append({
        'name': 'Flickr30k-Full',
        'dataset': 'nlphuji/flickr30k',
        'split': 'test',
        'max_samples': None
    })
    
    # COCO
    configurations.append({
        'name': 'COCO-Small',
        'dataset': 'laion/coco',
        'split': 'validation',
        'max_samples': 1000
    })
    
    # Conceptual Captions
    configurations.append({
        'name': 'Conceptual-Captions',
        'dataset': 'conceptual_captions',
        'split': 'validation',
        'max_samples': 1000
    })
    
    return configurations


def get_model_configurations():
    """
    Get model configurations for experiments
    
    Returns:
        List of model configurations
    """
    configurations = []
    
    # CLIP ViT-B/32
    configurations.append({
        'name': 'CLIP ViT-B/32',
        'model_name': 'openai/clip-vit-base-patch32',
        'model_class': CLIPModel,
        'processor_class': CLIPProcessor
    })
    
    # CLIP ViT-B/16
    configurations.append({
        'name': 'CLIP ViT-B/16',
        'model_name': 'openai/clip-vit-base-patch16',
        'model_class': CLIPModel,
        'processor_class': CLIPProcessor
    })
    
    # CLIP ViT-L/14
    configurations.append({
        'name': 'CLIP ViT-L/14',
        'model_name': 'openai/clip-vit-large-patch14',
        'model_class': CLIPModel,
        'processor_class': CLIPProcessor
    })
    
    # BLIP
    configurations.append({
        'name': 'BLIP',
        'model_name': 'Salesforce/blip-image-captioning-base',
        'model_class': BlipForImageTextRetrieval,
        'processor_class': BlipProcessor
    })
    
    return configurations


###############################
# Main Experiment Functions   #
###############################

def run_single_experiment(model_config, dataset_config, attack_config, defense_configs, num_samples=None):
    """
    Run a single experiment with one model, dataset, attack, and multiple defenses
    
    Args:
        model_config: Model configuration dictionary
        dataset_config: Dataset configuration dictionary
        attack_config: Attack configuration dictionary
        defense_configs: List of defense configurations
        num_samples: Number of samples to use (overrides dataset_config)
        
    Returns:
        Dictionary of results
    """
    # Set up logging
    logger.info(f"Starting experiment with {model_config['name']} on {dataset_config['name']}")
    logger.info(f"Attack: {attack_config['name']}")
    logger.info(f"Testing {len(defense_configs)} defense configurations")
    
    # Override number of samples if specified
    if num_samples is not None:
        dataset_config['max_samples'] = num_samples
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load model and processor
    logger.info(f"Loading model: {model_config['name']}")
    model = model_config['model_class'].from_pretrained(model_config['model_name']).to(device)
    processor = model_config['processor_class'].from_pretrained(model_config['model_name'])
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_config['name']}")
    dataset_name = dataset_config['dataset']
    hf_dataset = load_dataset(dataset_name)
    
    dataset = HFDatasetWrapper(
        hf_dataset, 
        split=dataset_config['split'], 
        max_samples=dataset_config['max_samples']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64,
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Generate adversarial examples
    adv_examples = generate_adversarial_examples(model, processor, dataloader, attack_config, device)
    
    # Run evaluation with each defense configuration
    all_results = []
    
    # Add baseline evaluation (no defense)
    logger.info("Evaluating baseline (no defense)")
    baseline_metrics, _ = evaluate_defense(
        model, processor, dataloader, adv_examples,
        defense=None, defense_name="No Defense", device=device
    )
    all_results.append(baseline_metrics)
    
    # Evaluate each defense configuration
    for defense_config in defense_configs:
        defense_name = defense_config['name']
        logger.info(f"Evaluating defense: {defense_name}")
        
        defense_metrics, batch_metrics = evaluate_defense(
            model, processor, dataloader, adv_examples,
            defense=defense_config['config'],
            defense_name=defense_name,
            device=device
        )
        
        # Add defense configuration to metrics
        defense_metrics['defense_config'] = defense_config['config']
        all_results.append(defense_metrics)
    
    # Create visualizations
    logger.info("Creating visualizations")
    create_defense_comparison_plot(all_results, k=1)
    create_defense_comparison_plot(all_results, k=5)
    create_recovery_plot(all_results)
    
    # Save results table
    create_results_table(all_results)
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_results


def run_experiment_suite(model_configs=None, dataset_configs=None, 
                        attack_configs=None, defense_configs=None,
                        num_samples=None):
    """
    Run a suite of experiments with different configurations
    
    Args:
        model_configs: List of model configurations (default: all)
        dataset_configs: List of dataset configurations (default: all)
        attack_configs: List of attack configurations (default: all)
        defense_configs: List of defense configurations (default: all)
        num_samples: Number of samples to use for each experiment
    """
    # Get default configurations if not provided
    if model_configs is None:
        model_configs = get_model_configurations()
    if dataset_configs is None:
        dataset_configs = get_dataset_configurations()
    if attack_configs is None:
        attack_configs = get_attack_configurations()
    if defense_configs is None:
        defense_configs = get_defense_configurations()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    for model_config in model_configs:
        model_name = model_config['name']
        all_results[model_name] = {}
        
        for dataset_config in dataset_configs:
            dataset_name = dataset_config['name']
            all_results[model_name][dataset_name] = {}
            
            for attack_config in attack_configs:
                attack_name = attack_config['name']
                logger.info(f"\nRunning experiment: {model_name} - {dataset_name} - {attack_name}")
                
                try:
                    results = run_single_experiment(
                        model_config, dataset_config, attack_config,
                        defense_configs, num_samples
                    )
                    all_results[model_name][dataset_name][attack_name] = results
                    
                    # Save intermediate results
                    results_file = os.path.join(
                        RESULTS_DIR,
                        f"results_{model_name}_{dataset_name}_{attack_name}.json"
                    )
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error in experiment {model_name} - {dataset_name} - {attack_name}: {str(e)}")
                    continue
    
    return all_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run tensor decomposition defense experiments')
    parser.add_argument('--models', nargs='+', help='Models to test (default: all)')
    parser.add_argument('--datasets', nargs='+', help='Datasets to test (default: all)')
    parser.add_argument('--attacks', nargs='+', help='Attacks to test (default: all)')
    parser.add_argument('--defenses', choices=['standard', 'comprehensive', 'multilayer', 'all'],
                      default='standard', help='Defense configurations to test')
    parser.add_argument('--samples', type=int, help='Number of samples to use')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load configurations from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        model_configs = [config['models'][m] for m in (args.models or config['models'].keys())]
        dataset_configs = [config['datasets'][d] for d in (args.datasets or config['datasets'].keys())]
        attack_configs = [config['attacks'][a] for a in (args.attacks or config['attacks'].keys())]
        defense_configs = get_defense_configurations(args.defenses)
    else:
        # Use default configurations
        model_configs = None if not args.models else [
            c for c in get_model_configurations() if c['name'] in args.models
        ]
        dataset_configs = None if not args.datasets else [
            c for c in get_dataset_configurations() if c['name'] in args.datasets
        ]
        attack_configs = None if not args.attacks else [
            c for c in get_attack_configurations() if c['name'] in args.attacks
        ]
        defense_configs = get_defense_configurations(args.defenses)
    
    # Run experiments
    results = run_experiment_suite(
        model_configs=model_configs,
        dataset_configs=dataset_configs,
        attack_configs=attack_configs,
        defense_configs=defense_configs,
        num_samples=args.samples
    )
    
    # Save final results
    results_file = os.path.join(RESULTS_DIR, "final_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All experiments completed. Results saved to {results_file}")