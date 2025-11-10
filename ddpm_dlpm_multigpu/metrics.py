"""
Metrics collection utilities for monitoring model training.
General-purpose functions that work with any PyTorch model.
"""

import torch
import numpy as np


def gather_training_stats(
    local_loss,
    local_grad_norm,
    local_grad_below_thresh,
    local_weight_norm,
    local_zero_weight_pct,
    local_update_ratio,
    grad_thresholds,
    rank,
    world_size,
    device
):
    """
    Gather training statistics from all distributed processes to rank 0.

    Each rank computes local statistics from its batches, then this function:
    - Gathers all statistics to rank 0
    - Averages them across all ranks
    - Returns global averages (only on rank 0, None on others)

    Args:
        local_loss: Mean loss for this rank's batches (scalar)
        local_grad_norm: Mean gradient norm for this rank (scalar)
        local_grad_below_thresh: Dict {threshold: percentage} for this rank
        local_weight_norm: Mean weight norm for this rank (scalar)
        local_zero_weight_pct: Mean zero weight percentage for this rank (scalar)
        local_update_ratio: Mean update ratio for this rank (scalar)
        grad_thresholds: List of gradient thresholds to track
        rank: Process rank in distributed group
        world_size: Total number of processes
        device: Device to use for tensors

    Returns:
        If rank == 0:
            dict with keys: 'loss', 'grad_norm', 'grad_below_thresh',
                           'weight_norm', 'zero_weight_pct', 'update_ratio'
        If rank != 0:
            None
    """
    import torch.distributed as dist

    # Convert to tensors for gathering
    loss_tensor = torch.tensor(local_loss, device=device)
    grad_norm_tensor = torch.tensor(local_grad_norm, device=device)
    grad_below_thresh_tensors = {thresh: torch.tensor(val, device=device) for thresh, val in local_grad_below_thresh.items()}
    weight_norm_tensor = torch.tensor(local_weight_norm, device=device)
    zero_weight_pct_tensor = torch.tensor(local_zero_weight_pct, device=device)
    update_ratio_tensor = torch.tensor(local_update_ratio, device=device)

    # Prepare gather lists on rank 0
    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)] if rank == 0 else None
    gathered_grad_norms = [torch.zeros_like(grad_norm_tensor) for _ in range(world_size)] if rank == 0 else None
    gathered_grad_below_thresh = {
        thresh: [torch.zeros_like(grad_below_thresh_tensors[thresh]) for _ in range(world_size)] if rank == 0 else None
        for thresh in grad_thresholds
    }
    gathered_weight_norms = [torch.zeros_like(weight_norm_tensor) for _ in range(world_size)] if rank == 0 else None
    gathered_zero_weight_pcts = [torch.zeros_like(zero_weight_pct_tensor) for _ in range(world_size)] if rank == 0 else None
    gathered_update_ratios = [torch.zeros_like(update_ratio_tensor) for _ in range(world_size)] if rank == 0 else None

    # Gather all statistics to rank 0
    dist.gather(loss_tensor, gathered_losses, dst=0)
    dist.gather(grad_norm_tensor, gathered_grad_norms, dst=0)
    for thresh in grad_thresholds:
        dist.gather(grad_below_thresh_tensors[thresh], gathered_grad_below_thresh[thresh], dst=0)
    dist.gather(weight_norm_tensor, gathered_weight_norms, dst=0)
    dist.gather(zero_weight_pct_tensor, gathered_zero_weight_pcts, dst=0)
    dist.gather(update_ratio_tensor, gathered_update_ratios, dst=0)

    # Only rank 0 computes global averages
    if rank == 0:
        # Average across all ranks
        global_loss = torch.stack(gathered_losses).mean().item()
        global_grad_norm = torch.stack(gathered_grad_norms).mean().item()
        global_grad_below_thresh = {
            thresh: torch.stack(gathered_grad_below_thresh[thresh]).mean().item()
            for thresh in grad_thresholds
        }
        global_weight_norm = torch.stack(gathered_weight_norms).mean().item()
        global_zero_weight_pct = torch.stack(gathered_zero_weight_pcts).mean().item()
        global_update_ratio = torch.stack(gathered_update_ratios).mean().item()

        return {
            'loss': global_loss,
            'grad_norm': global_grad_norm,
            'grad_below_thresh': global_grad_below_thresh,
            'weight_norm': global_weight_norm,
            'zero_weight_pct': global_zero_weight_pct,
            'update_ratio': global_update_ratio
        }
    else:
        return None


def collect_gradient_stats(model, thresholds=None):
    """
    Collect gradient statistics from model parameters.

    Args:
        model: PyTorch model with computed gradients
        thresholds: List of thresholds to track gradient sparsity at.
                   If None, uses default [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]

    Returns:
        dict: Dictionary containing gradient statistics
            - grad_norm: L2 norm of all gradients
            - grad_below_thresh: Dict mapping threshold to percentage of gradients below it
    """
    if thresholds is None:
        thresholds = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]

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
