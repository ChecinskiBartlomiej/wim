import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
import torch.distributed as dist


class InceptionFeatureExtractor(torch.nn.Module):
    """
    InceptionV3 feature extractor for FID calculation.
    Extracts 2048-dimensional features from the pool3 layer.
    """

    def __init__(self, device='cpu', pretrained_path=None):
        super().__init__()

        # Load InceptionV3 model
        print(f"Loading InceptionV3 from: {pretrained_path}")
        inception = inception_v3(weights=None, transform_input=False)
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        inception.load_state_dict(state_dict)

        # Change classification to identity
        inception.eval()
        inception.fc = torch.nn.Identity()

        # Move model to target device
        self.inception = inception.to(device)

        # Freeze parameters
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

        # InceptionV3 expects 3 channels - convert grayscale to RGB if needed by replicating greyscale channel to 3 channels (R=G=B)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Get features - output is already [B, 2048]
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

    # Calculate fid
    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    fid = mean_diff + trace_term

    return fid


def extract_inception_features(images, batch_size=512, device='cpu', pretrained_path=None):
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

    # Extract features
    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch_images = batch[0].to(device)
            features = feature_extractor(batch_images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def load_real_images_for_fid(dataset, num_images, batch_size=128, start_idx=0):
    """
    Load real images from dataset for FID calculation.

    Args:
        dataset: PyTorch dataset (returns images in [-1, 1])
        num_images: Number of images to load
        batch_size: Batch size for loading (default: 128)
        start_idx: Starting index in dataset (default: 0, for distributed loading)

    Returns:
        images: Tensor [B, C, H, W] in range [-1, 1]
    """
    num_to_load = min(num_images, len(dataset) - start_idx)

    # Create subset of dataset (from start_idx to start_idx + num_to_load)
    subset = Subset(dataset, range(start_idx, start_idx + num_to_load))

    # Load images in batches
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
        images: Tensor [B, C, H, W] in range [-1, 1]
    """
    batch_size = cfg.fid_batch_size
    generated_images = []

    # Calculate number of full batches and remainder
    num_full_batches = num_images // batch_size
    remainder = num_images % batch_size

    # Generate full batches
    for i in tqdm(range(num_full_batches), desc="Generating images for FID"):
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


def calculate_fid_from_model_distributed(diffusion, model, dataset, cfg, rank, world_size, device='cpu'):
    """
    Calculate FID score using distributed generation and feature extraction across multiple GPUs.

    This function parallelizes FID calculation:
    - Each GPU loads a portion of real images
    - Each GPU generates num_fid_images // world_size images
    - Each GPU extracts features from its generated images
    - All features are gathered to rank 0 for FID computation

    Args:
        diffusion: Diffusion process (DDPM or DLPM)
        model: Trained U-Net model (should NOT be wrapped in DDP)
        dataset: PyTorch dataset containing real images
        cfg: Configuration object (must have num_fid_images and inception_path)
        rank: Process rank in distributed group
        world_size: Total number of processes
        device: Device to use (should be f'cuda:{rank}')

    Returns:
        fid_score: FID score (scalar, lower is better) - only valid on rank 0, None on other ranks
    """
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"CALCULATING FID SCORE (DISTRIBUTED)")
        print(f"{'='*70}\n")
        print(f"Using {world_size} GPUs for parallel generation and feature extraction")

    # Calculate how many images each GPU should process
    images_per_gpu = cfg.num_fid_images // world_size
    remainder = cfg.num_fid_images % world_size

    # Rank 0 handles the remainder
    if rank == 0:
        my_num_images = images_per_gpu + remainder
        print(f"Rank 0 will generate {my_num_images} images (includes remainder)")
        print(f"Other ranks will generate {images_per_gpu} images each")
        print(f"Total: {cfg.num_fid_images} images\n")
    else:
        my_num_images = images_per_gpu

    # Load real images - each rank loads its portion
    start_idx = rank * images_per_gpu + (remainder if rank > 0 else 0)
    end_idx = start_idx + my_num_images

    if rank == 0:
        print(f"Loading real images (rank {rank}: indices {start_idx} to {end_idx})...")

    real_images = load_real_images_for_fid(dataset, my_num_images, start_idx=start_idx).to(device)

    if rank == 0:
        print(f"Rank {rank} loaded {len(real_images)} real images")

    # Generate images from model - each rank generates its portion
    if rank == 0:
        print(f"\nRank {rank} generating {my_num_images} images...")

    model.eval()
    with torch.no_grad():
        generated_images = generate_images_for_fid(diffusion, model, cfg, my_num_images)
        generated_images = generated_images.to(device)

    if rank == 0:
        print(f"Rank {rank} generated images: {generated_images.shape}")

    # Extract features from real and generated images on each GPU
    if rank == 0:
        print(f"\nRank {rank} extracting features from real images...")

    real_features = extract_inception_features(
        real_images,
        batch_size=512,
        device=device,
        pretrained_path=str(cfg.inception_path)
    )

    if rank == 0:
        print(f"Rank {rank} extracting features from generated images...")

    gen_features = extract_inception_features(
        generated_images,
        batch_size=512,
        device=device,
        pretrained_path=str(cfg.inception_path)
    )

    if rank == 0:
        print(f"Rank {rank} extracted features: real={real_features.shape}, gen={gen_features.shape}")

    # Convert to tensors for gathering
    real_features_tensor = torch.from_numpy(real_features).to(device)
    gen_features_tensor = torch.from_numpy(gen_features).to(device)

    # Synchronize all ranks before gathering
    dist.barrier()

    # Gather all features to rank 0
    if rank == 0:
        print(f"\nGathering features from all {world_size} GPUs to rank 0...")

        # Prepare lists to receive features from all ranks
        all_real_features = [torch.zeros_like(real_features_tensor) for _ in range(world_size)]
        all_gen_features = [torch.zeros_like(gen_features_tensor) for _ in range(world_size)]

        # Adjust size for rank 0 if it has remainder
        if remainder > 0:
            all_real_features[0] = torch.zeros(my_num_images, real_features_tensor.shape[1], device=device)
            all_gen_features[0] = torch.zeros(my_num_images, gen_features_tensor.shape[1], device=device)
    else:
        all_real_features = None
        all_gen_features = None

    # Gather features
    dist.gather(real_features_tensor, all_real_features if rank == 0 else None, dst=0)
    dist.gather(gen_features_tensor, all_gen_features if rank == 0 else None, dst=0)

    # Only rank 0 calculates FID
    if rank == 0:
        # Concatenate all features
        all_real_features_concat = torch.cat(all_real_features, dim=0).cpu().numpy()
        all_gen_features_concat = torch.cat(all_gen_features, dim=0).cpu().numpy()

        print(f"Combined features: real={all_real_features_concat.shape}, gen={all_gen_features_concat.shape}")

        # Calculate statistics for real images
        mu_real = np.mean(all_real_features_concat, axis=0)
        sigma_real = np.cov(all_real_features_concat, rowvar=False)

        # Calculate statistics for generated images
        mu_gen = np.mean(all_gen_features_concat, axis=0)
        sigma_gen = np.cov(all_gen_features_concat, rowvar=False)

        # Calculate FID
        print("Computing Fréchet distance...")
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        print(f"\n{'='*70}")
        print(f"FID Score: {fid_score:.2f}")
        print(f"{'='*70}\n")

        return fid_score
    else:
        return None
