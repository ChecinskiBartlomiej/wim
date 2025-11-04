"""
Metrics collection utilities for monitoring model training.
General-purpose functions that work with any PyTorch model.
"""

import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from pathlib import Path


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


class InceptionFeatureExtractor(torch.nn.Module):
    """
    InceptionV3 feature extractor for FID calculation.
    Extracts 2048-dimensional features from the pool3 layer.
    """

    def __init__(self, device='cpu', pretrained_path=None):
        super().__init__()

        print(f"Loading InceptionV3 from: {pretrained_path}")
        inception = inception_v3(weights=None, transform_input=False)
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        inception.load_state_dict(state_dict)
        inception.eval()

        inception.fc = torch.nn.Identity()

        self.inception = inception.to(device)

        for param in self.inception.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extract features from input images.

        Args:
            x: Batch of images [B, C, 299, 299] in range [-1, 1]
               where C=1 (grayscale) or C=3 (RGB)

        Returns:
            features: [B, 2048] feature vectors
        """
        # InceptionV3 expects images in range [0, 1], so convert from [-1, 1]
        x = (x + 1) / 2

        # InceptionV3 expects 3 channels - convert grayscale to RGB if needed
        if x.shape[1] == 1:
            # Replicate grayscale channel to 3 channels (R=G=B)
            x = x.repeat(1, 3, 1, 1)

        # Get features - output is already [B, 2048], no squeeze needed!
        features = self.inception(x)

        return features


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two Gaussian distributions.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1: Mean of first distribution [2048]
        sigma1: Covariance of first distribution [2048, 2048]
        mu2: Mean of second distribution [2048]
        sigma2: Covariance of second distribution [2048, 2048]
        eps: Small constant for numerical stability

    Returns:
        fid: Fréchet distance (scalar)
    """
    
    diff = mu1 - mu2
    mean_diff = np.sum(diff ** 2)

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Check for imaginary components (numerical errors)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = covmean.real

    # Calculate trace term
    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    fid = mean_diff + trace_term

    return fid


def extract_inception_features(images, batch_size=50, device='cpu', pretrained_path=None):
    """
    Extract InceptionV3 features from a batch of images.

    Args:
        images: Tensor of images [B, C, H, W] in range [-1, 1]
        batch_size: Batch size for processing
        device: Device to use ('cpu' or 'cuda')
        pretrained_path: Path to pretrained InceptionV3 weights (for offline use)

    Returns:
        features: [B, 2048] feature array
    """
    feature_extractor = InceptionFeatureExtractor(device=device, pretrained_path=pretrained_path)

    # Resize images to 299x299 (InceptionV3 input size)
    if images.shape[-1] != 299:
        images = torch.nn.functional.interpolate(
            images,
            size=(299, 299),
            mode='bilinear',
            align_corners=False
        )

    # Create dataloader
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch_images = batch[0].to(device)
            features = feature_extractor(batch_images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def load_real_images_for_fid(dataset, num_images, batch_size=128):
    """
    Load real images from dataset for FID calculation.

    Args:
        dataset: PyTorch dataset (returns images in [-1, 1])
        num_images: Number of images to load
        batch_size: Batch size for loading (default: 128)

    Returns:
        images: Tensor [B, C, H, W] in range [-1, 1]
    """
    num_to_load = min(num_images, len(dataset))

    # Create subset of dataset (first num_to_load images)
    subset = Subset(dataset, range(num_to_load))

    # Use DataLoader for efficient batched loading
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    real_images = []
    for batch in tqdm(dataloader, desc="Loading real images"):
        real_images.append(batch)

    return torch.cat(real_images, dim=0)


def generate_images_for_fid(diffusion, model, cfg, num_images):
    """
    Generate images from diffusion model for FID calculation.

    Args:
        diffusion: Diffusion process (DDPM or DLPM)
        model: Trained U-Net model (should be in eval mode)
        cfg: Configuration object (must have fid_batch_size attribute)
        num_images: Number of images to generate

    Returns:
        images: Tensor [B, C, H, W] in range [-1, 1] where B = num_images
    """
    batch_size = cfg.fid_batch_size
    generated_images = []

    # Calculate number of full batches and remainder
    num_full_batches = num_images // batch_size
    remainder = num_images % batch_size

    # Generate full batches
    for i in tqdm(range(num_full_batches), desc="Generating images for FID"):
        # Generate batch (returns tensor [batch_size, C, H, W] in range [-1, 1])
        samples = diffusion.generate_samples(cfg, model, return_intermediate=False, batch_size=batch_size)

        # Split batch into individual images: [batch_size, C, H, W] -> list of [C, H, W]
        for j in range(batch_size):
            generated_images.append(samples[j])

    # Generate remaining images if any
    if remainder > 0:
        samples = diffusion.generate_samples(cfg, model, return_intermediate=False, batch_size=remainder)
        for j in range(remainder):
            generated_images.append(samples[j])

    return torch.stack(generated_images)

def calculate_fid(real_images, generated_images, batch_size=50, device='cpu', pretrained_path=None):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_images: Tensor of real images [B, C, H, W] in range [-1, 1]
                    where C=1 (grayscale/MNIST) or C=3 (RGB/CIFAR10)
        generated_images: Tensor of generated images [B, C, H, W] in range [-1, 1]
        batch_size: Batch size for feature extraction (default: 50)
        device: Device to use ('cpu' or 'cuda')
        pretrained_path: Path to pretrained InceptionV3 weights (for offline use).
                        Example: 'pretrained_models/inception_v3_imagenet.pth'

    Returns:
        fid_score: FID score (scalar, lower is better)
    """
    print(f"Calculating FID with {len(real_images)} real and {len(generated_images)} generated images...")

    # Extract features from real images
    print("Extracting features from real images...")
    real_features = extract_inception_features(real_images, batch_size, device, pretrained_path)

    # Extract features from generated images
    print("Extracting features from generated images...")
    gen_features = extract_inception_features(generated_images, batch_size, device, pretrained_path)

    # Calculate statistics for real images
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    # Calculate statistics for generated images
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # Calculate FID
    print("Computing Fréchet distance...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    print(f"FID Score: {fid_score:.2f}")

    return fid_score


def calculate_fid_from_model(diffusion, model, dataset, cfg, device='cpu'):
    """
    Calculate FID score by generating images from a model and comparing to real dataset.

    This is a high-level convenience function that handles:
    - Loading real images from dataset
    - Generating images from the model
    - Computing FID score

    Args:
        diffusion: Diffusion process (DDPM or DLPM)
        model: Trained U-Net model
        dataset: PyTorch dataset containing real images
        cfg: Configuration object (must have num_fid_images and inception_path)
        device: Device to use ('cpu' or 'cuda')

    Returns:
        fid_score: FID score (scalar, lower is better)
    """
    print(f"\n{'='*70}")
    print(f"CALCULATING FID SCORE")
    print(f"{'='*70}\n")

    # Load real images
    real_images = load_real_images_for_fid(dataset, cfg.num_fid_images)
    print(f"Loaded real images: {real_images.shape}")

    # Generate images from model
    model.eval()
    with torch.no_grad():
        generated_images = generate_images_for_fid(diffusion, model, cfg, cfg.num_fid_images)
    print(f"Generated images: {generated_images.shape}")

    # Calculate FID
    fid_score = calculate_fid(
        real_images,
        generated_images,
        batch_size=50,
        device=device,
        pretrained_path=str(cfg.inception_path)
    )

    print(f"\n{'='*70}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'='*70}\n")

    return fid_score



