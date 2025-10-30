"""
Metrics collection utilities for monitoring model training.
General-purpose functions that work with any PyTorch model.
"""

import torch
import numpy as np


def collect_gradient_stats(model):
    """
    Collect gradient statistics from model parameters.

    Args:
        model: PyTorch model with computed gradients

    Returns:
        dict: Dictionary containing gradient statistics
            - grad_norm: L2 norm of all gradients
            - zero_grad_pct: Percentage of near-zero gradients
    """
    all_grads = []

    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.flatten())

    if len(all_grads) == 0:
        return {"grad_norm": 0.0, "zero_grad_pct": 0.0}

    all_grads = torch.cat(all_grads)

    grad_norm = torch.norm(all_grads).item()
    zero_grad_pct = (torch.sum(torch.abs(all_grads) < 1e-7).item() / all_grads.numel()) * 100

    return {
        "grad_norm": grad_norm,
        "zero_grad_pct": zero_grad_pct
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
            ratio = torch.abs(weight_update) / (torch.abs(old_weights[name]) + 1e-10)
            update_ratios.append(ratio.flatten())

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
        all_update_ratios = torch.cat(update_ratios)
        result["update_ratio"] = torch.mean(all_update_ratios).item()
    else:
        result["update_ratio"] = 0.0

    return result
