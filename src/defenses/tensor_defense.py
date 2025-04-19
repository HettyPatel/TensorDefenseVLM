"""
Tensor decomposition defense for Vision-Language Models.

This module implements tensor decomposition methods (CP, Tucker, Tensor Train)
applied to specific layers of vision encoders to defend against adversarial attacks.
"""

import torch
import logging
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train
from .base_defense import BaseTensorDefense

logger = logging.getLogger("tensor_defense")

class TargetedTensorDefense(BaseTensorDefense):
    """Targeted tensor decomposition defense for Vision-Language Models
    
    This class applies tensor decomposition to specific layers of a vision encoder
    to defend against adversarial attacks on images. By decomposing and reconstructing
    intermediate representations, the defense aims to remove adversarial perturbations
    while preserving semantic content.
    """
    def __init__(self, model, method='cp', rank=64, alpha=0.5, 
                 target_layer='final_norm', vision_layer_idx=-1):
        """
        Initialize the tensor decomposition defense
        
        Args:
            model: VLM model to defend (CLIP or LLaVA)
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
        """Register forward hooks on the targeted layer based on model architecture"""
        # Determine model type and structure
        if hasattr(self.model, 'vision_model'):
            # CLIP model
            self._register_clip_hooks()
        elif hasattr(self.model, 'vision_encoder'):
            # BLIP model
            self._register_blip_hooks()
        elif hasattr(self.model, 'vision_tower'):
            # LLaVA model
            self._register_llava_hooks()
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
    
    def _register_llava_hooks(self):
        """Register hooks for LLaVA model architecture"""
        try:
            # Access the underlying vision tower (usually a CLIP vision model)
            vision_model = self.model.vision_tower.vision_model
            
            # Get total number of vision layers for negative indexing
            num_vision_layers = len(vision_model.encoder.layers)
            
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
                # Final layer norm after the MLP
                vision_module = vision_model.encoder.layers[layer_idx].layer_norm2
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on LLaVA vision encoder layer {layer_idx} final norm")
            
            elif self.target_layer == 'attention':
                # Self-attention output
                vision_module = vision_model.encoder.layers[layer_idx].self_attn.out_proj
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on LLaVA vision encoder layer {layer_idx} attention output")
            
            elif self.target_layer == 'mlp':
                # MLP output (fc2)
                vision_module = vision_model.encoder.layers[layer_idx].mlp.fc2
                vision_hook = vision_module.register_forward_hook(self._forward_hook)
                self.hook_handles.append(vision_hook)
                logger.info(f"Registered hook on LLaVA vision encoder layer {layer_idx} MLP output")
        
        except (AttributeError, IndexError) as e:
            logger.error(f"Error setting up LLaVA hooks: {str(e)}")
    
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
                # Reshape to 3D tensor for CP decomposition if needed
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