"""
Projected Gradient Descent (PGD) attack for Vision-Language Models.

This module implements PGD attacks against CLIP and LLaVA models
by minimizing the cosine similarity between image and text embeddings.
"""

import torch
import logging

logger = logging.getLogger("tensor_defense")

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
            model: VLM model to attack (CLIP or LLaVA)
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
            Tuple containing:
            - Tensor of adversarial examples
            - Processed inputs
            - Original pixel values
        """
        # Determine model type to handle different model architectures
        is_clip = hasattr(self.model, 'get_image_features')
        is_llava = hasattr(self.model, 'vision_tower')
        
        # Convert PIL images to tensors using the processor
        if is_clip:
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
        elif is_llava:
            # LLaVA processors work differently, typically separate processors for image and text
            if hasattr(self.processor, 'image_processor'):
                # If processor has separate image and text processors
                vision_inputs = self.processor.image_processor(images, return_tensors="pt").to(device)
                text_inputs = self.processor.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
                inputs = {**vision_inputs, **text_inputs}
            else:
                # Fallback for other processor types
                inputs = self.processor(images, captions, return_tensors="pt", padding=True, truncation=True).to(device)
                
            # Handle targeted attack for LLaVA
            if self.targeted and target_captions is not None:
                if hasattr(self.processor, 'tokenizer'):
                    target_inputs = self.processor.tokenizer(
                        target_captions, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
                else:
                    target_inputs = self.processor(
                        None, target_captions, return_tensors="pt", padding=True, truncation=True
                    ).to(device)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        
        # Get the pixel values tensor (original images)
        pixel_values = inputs['pixel_values'].clone().detach() if 'pixel_values' in inputs else inputs['images'].clone().detach()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize adversarial examples with original pixel values
        adv_images = pixel_values.clone().detach()
        
        # Random initialization for PGD (if enabled)
        if self.random_start and self.steps > 1:
            # Add small random noise to start
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon/2, self.epsilon/2)
            adv_images = torch.clamp(adv_images, 0, 1)
        
        # Perform multi-step attack
        for step in range(self.steps):
            # Create a fresh copy that requires gradients
            adv_images = adv_images.clone().detach().requires_grad_(True)
            
            # Forward pass with gradient computation enabled
            with torch.enable_grad():
                # CLIP-specific attack logic
                if is_clip:
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
                
                # LLaVA-specific attack logic
                elif is_llava:
                    # Update inputs with adversarial images
                    adv_inputs = inputs.copy()
                    if 'pixel_values' in adv_inputs:
                        adv_inputs['pixel_values'] = adv_images
                    else:
                        adv_inputs['images'] = adv_images
                    
                    # Forward pass through LLaVA
                    outputs = self.model(**adv_inputs)
                    
                    # For LLaVA, we need to extract image and text embeddings differently
                    # The specific approach depends on the LLaVA implementation
                    if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
                        # If embeddings are directly available
                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                        
                        # Normalize embeddings
                        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                        
                        # Compute similarity
                        similarity = torch.sum(image_embeds * text_embeds, dim=-1)
                        
                        # Loss is negative similarity (we want to minimize similarity)
                        loss = -similarity.sum()
                    else:
                        # Fallback to using model logits or hidden states
                        # This depends on the specific LLaVA architecture
                        if hasattr(outputs, 'logits'):
                            # Use model logits as a proxy for alignment
                            loss = -outputs.logits.mean()
                        else:
                            # Use the last hidden states as a proxy
                            loss = -outputs.last_hidden_state.mean()
            
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