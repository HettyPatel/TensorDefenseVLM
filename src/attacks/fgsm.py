"""
Fast Gradient Sign Method (FGSM) attack for Vision-Language Models.

This module implements a simple one-step gradient-based attack that
adds perturbations to maximize the loss.
"""

import torch
import logging
from .pgd import PGDAttack

logger = logging.getLogger("tensor_defense")

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
            model: VLM model to attack (CLIP or LLaVA)
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
        # FGSM is essentially PGD with a single step, so we leverage the PGD implementation
        pgd_attack = PGDAttack(
            model=self.model,
            processor=self.processor,
            epsilon=self.epsilon,
            alpha=self.epsilon,  # Step size equals epsilon for one-step attack
            steps=1,
            random_start=False
        )
        
        return pgd_attack.perturb(images, captions, device)