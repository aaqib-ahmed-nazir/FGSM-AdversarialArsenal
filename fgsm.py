import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    loss_fn: Optional[nn.Module] = None,
    targeted: bool = False,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    Performs the Fast Gradient Sign Method (FGSM) attack on the input images.
    
    Args:
        model: The model to attack
        images: Input images
        labels: True labels for untargeted attack, target labels for targeted attack
        epsilon: Attack strength parameter
        loss_fn: Loss function to use (defaults to CrossEntropyLoss if None)
        targeted: If True, perform targeted attack; otherwise, untargeted attack
        clamp_min: Minimum value for clamping the adversarial example
        clamp_max: Maximum value for clamping the adversarial example
        
    Returns:
        Adversarial examples
    """
    model.eval()
    
    perturbed_images = images.clone().detach().requires_grad_(True)
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model(perturbed_images)
    
    # Calculate loss
    loss = loss_fn(outputs, labels)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Calculate gradients of model in backward pass
    loss.backward()
    
    # Collect the element-wise sign of the data gradient
    data_grad = perturbed_images.grad.data
    
    # Create the perturbed image by adjusting each pixel based on the sign of the gradient
    if targeted:
        perturbed_images = perturbed_images - epsilon * torch.sign(data_grad)
    
    # For untargeted attacks, any random direction might work
    else:
        perturbed_images = perturbed_images + epsilon * torch.sign(data_grad)
    
    # Clipping to maintain the range
    perturbed_images = torch.clamp(perturbed_images, clamp_min, clamp_max)
    
    return perturbed_images


def get_model_prediction(
    model: nn.Module, 
    images: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the model's prediction on the input images.
    
    Args:
        model: The model to use for prediction
        images: Input images
        
    Returns:
        Tuple of (class indices, probability scores)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probs, dim=1)
    
    return predicted_classes, probs


def evaluate_attack_success(
    model: nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Evaluate the success rate of the attack.
    
    Args:
        model: The model to evaluate
        original_images: Original clean images
        adversarial_images: Adversarial examples
        true_labels: Ground truth labels
        
    Returns:
        Tuple of (original accuracy, adversarial accuracy, attack success rate)
    """
    model.eval()
    
    # Get predictions for original images
    orig_pred, _ = get_model_prediction(model, original_images)
    orig_correct = (orig_pred == true_labels).sum().item()
    orig_accuracy = orig_correct / float(len(true_labels))
    
    # Get predictions for adversarial images
    adv_pred, _ = get_model_prediction(model, adversarial_images)
    adv_correct = (adv_pred == true_labels).sum().item()
    adv_accuracy = adv_correct / float(len(true_labels))
    
    # Calculate attack success rate
    originally_correct = (orig_pred == true_labels)
    now_incorrect = (adv_pred != true_labels)
    attack_success = (originally_correct & now_incorrect).sum().item() / originally_correct.sum().item()
    
    return orig_accuracy, adv_accuracy, attack_success
