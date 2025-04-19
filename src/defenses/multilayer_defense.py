"""
Multi-layer tensor decomposition defense for Vision-Language Models.

This module implements a defense that applies tensor decomposition to
multiple layers simultaneously for enhanced protection against adversarial attacks.
"""

import logging
from .base_defense import BaseTensorDefense
from .tensor_defense import TargetedTensorDefense

logger = logging.getLogger("tensor_defense")

class MultiLayerTensorDefense(BaseTensorDefense):
    """Multi-layer tensor decomposition defense
    
    This class applies tensor decomposition to multiple layers 
    simultaneously to provide stronger defense against adversarial attacks.
    """
    
    def __init__(self, model, layer_configs):
        """
        Initialize the multi-layer tensor defense
        
        Args:
            model: VLM model to defend (CLIP or LLaVA)
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