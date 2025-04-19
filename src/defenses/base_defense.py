"""Base class for tensor decomposition defense strategies"""

import logging
import torch
import tensorly as tl

logger = logging.getLogger("tensor_defense")

class BaseTensorDefense:
    """Base class for tensor decomposition defense strategies
    
    This class provides the foundation for all tensor decomposition defense
    mechanisms against adversarial attacks on Vision-Language Models.
    """
    
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