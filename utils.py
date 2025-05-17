import torch
import os
import random
import numpy as np

def calculate_accuracy(y_pred, y_true):
    """
    Calculates classification accuracy.
    Args:
        y_pred (torch.Tensor): Raw logits or probabilities. Shape: (batch_size, num_classes)
        y_true (torch.Tensor): Ground truth labels. Shape: (batch_size)
    Returns:
        float: Accuracy.
    """
    if y_pred.ndim == 2:
        predicted_labels = torch.argmax(y_pred, dim=1)
    elif y_pred.ndim == 1: # Assuming already class indices
        predicted_labels = y_pred
    else:
        raise ValueError(f"Unsupported y_pred dimensions: {y_pred.ndim}")

    correct = (predicted_labels == y_true).sum().item()
    total = y_true.size(0)
    if total == 0:
        return 0.0
    return correct / total

def get_grl_alpha_dann(current_iter, total_iters, gamma=10.0, max_alpha=1.0, adapt_iter_offset=0):
    """
    Calculate alpha for GRL based on iteration progress (DANN-style).
    Allows an offset for when adaptation starts.
    Args:
        current_iter: Current training iteration.
        total_iters: Total training iterations for the adversarial phase.
        gamma: Controls the steepness of the schedule.
        max_alpha: Maximum value for alpha.
        adapt_iter_offset: Iteration number when GRL alpha starts increasing from 0.
    """
    adjusted_iter = max(0, current_iter - adapt_iter_offset)
    adjusted_total_iters = max(1, total_iters - adapt_iter_offset) # Avoid division by zero

    if adjusted_iter == 0: # Before adaptation starts or at the very beginning
        return 0.0
        
    p = float(adjusted_iter) / float(adjusted_total_iters)
    alpha_val = (2. / (1. + np.exp(-gamma * p))) - 1.0
    return np.clip(alpha_val * max_alpha, 0, max_alpha)


def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # Disabling benchmark and enabling deterministic may slow down training
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    print(f"Random seed set to {seed}")


if __name__ == '__main__':
    y_pred_logits = torch.tensor([[0.1, 2.5, 0.3], [3.0, 0.1, 0.1], [0.2, 0.3, 4.0], [1.0, 0.9, 0.8]])
    y_true_labels = torch.tensor([1, 0, 1, 0])
    accuracy = calculate_accuracy(y_pred_logits, y_true_labels)
    print(f"Calculated Accuracy: {accuracy}")

    # Test GRL alpha scheduling
    total_iters_test = 1000
    for i in range(0, total_iters_test + 100, 100):
        print(f"Iter {i}/{total_iters_test}: Alpha = {get_grl_alpha_dann(i, total_iters_test, adapt_iter_offset=200):.4f}")

    seed_everything(123)
