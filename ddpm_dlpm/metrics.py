"""
Metrics collection utilities for monitoring model training.
General-purpose functions that work with any PyTorch model.
"""

import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def collect_gradient_stats(model, thresholds=None):
    """
    Collect gradient statistics from model parameters.

    Args:
        model: PyTorch model with computed gradients
        thresholds: List of thresholds to track gradient sparsity at.
                   If None, uses default [1e-10, 1e-8, 1e-6, 1e-4]

    Returns:
        dict: Dictionary containing gradient statistics
            - grad_norm: L2 norm of all gradients
            - grad_below_thresh: Dict mapping threshold to percentage of gradients below it
    """
    if thresholds is None:
        thresholds = [10e-11, 10e-10, 10e-9, 10e-8, 10e-7]

    all_grads = []

    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.flatten())

    if len(all_grads) == 0:
        result = {"grad_norm": 0.0, "grad_below_thresh": {}}
        for thresh in thresholds:
            result["grad_below_thresh"][thresh] = 0.0
        return result

    all_grads = torch.cat(all_grads)
    abs_grads = torch.abs(all_grads)
    total_grads = all_grads.numel()

    grad_norm = torch.norm(all_grads).item()

    # Calculate percentage below each threshold
    grad_below_thresh = {}
    for thresh in thresholds:
        pct = (torch.sum(abs_grads < thresh).item() / total_grads) * 100
        grad_below_thresh[thresh] = pct

    return {
        "grad_norm": grad_norm,
        "grad_below_thresh": grad_below_thresh
    }


def save_weights_snapshot(model):
    """
    Save current model weights for later comparison (e.g., update ratio).

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary mapping parameter names to cloned weight tensors
    """
    weights_snapshot = {}
    for name, param in model.named_parameters():
        weights_snapshot[name] = param.data.clone()
    return weights_snapshot


def collect_weight_stats(model, old_weights=None):
    """
    Collect weight statistics from model parameters.

    Args:
        model: PyTorch model
        old_weights: Optional dictionary of previous weights (from save_weights_snapshot)
                    If provided, computes update ratio

    Returns:
        dict: Dictionary containing weight statistics
            - weight_norm: L2 norm of all weights
            - zero_weight_pct: Percentage of near-zero weights
            - update_ratio: Mean ratio of weight updates (if old_weights provided)
    """
    all_weights = []
    update_ratios = []

    for name, param in model.named_parameters():
        weight = param.data.flatten()
        all_weights.append(weight)

        # Compute update ratio if old weights are provided
        if old_weights is not None and name in old_weights:
            weight_update = param.data - old_weights[name]
            update_norm = torch.norm(weight_update).item()
            weight_norm = torch.norm(old_weights[name]).item()
            ratio = update_norm / (weight_norm + 1e-10)
            update_ratios.append(ratio)

    if len(all_weights) == 0:
        return {"weight_norm": 0.0, "zero_weight_pct": 0.0, "update_ratio": 0.0}

    all_weights = torch.cat(all_weights)

    weight_norm = torch.norm(all_weights).item()
    zero_weight_pct = (torch.sum(torch.abs(all_weights) < 1e-7).item() / all_weights.numel()) * 100

    result = {
        "weight_norm": weight_norm,
        "zero_weight_pct": zero_weight_pct
    }

    # Add update ratio if computed
    if len(update_ratios) > 0:
        result["update_ratio"] = np.mean(update_ratios)
    else:
        result["update_ratio"] = 0.0

    return result


def count_parameters(model):
    """
    Count the total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary containing parameter counts
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }
