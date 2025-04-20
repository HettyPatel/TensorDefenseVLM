"""
Base class for tensor decomposition defense strategies
"""

import torch
import logging
import tensorly as tl
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tensor_defense")

class BaseTensorDefense(ABC):
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
    
    @abstractmethod
    def _register_hooks(self):
        """Register forward hooks on the targeted layer(s)"""
        pass
    
    @abstractmethod
    def _forward_hook(self, module, input_tensor, output_tensor):
        """Hook function that applies tensor decomposition to layer outputs"""
        pass
    
    def __del__(self):
        """Cleanup when object is deleted"""
        self.remove_hooks()