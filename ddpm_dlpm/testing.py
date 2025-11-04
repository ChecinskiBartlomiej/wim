"""Testing script to generate images from trained DDPM models"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ddpm_dlpm.ddpm_cifar10.config import CONFIG
from ddpm_dlpm.unet import Unet


def plot_generated_images_nearest(images, save_path, cmap, im_channels, img_size):
    """
    Plot a grid of generated images with nearest-neighbor interpolation.

    Args:
        images: List of numpy arrays (flattened 1D images)
        save_path: Path to save the plot (should be Path object or string)
        cmap: Colormap for images ("gray" for grayscale, None for RGB)
        im_channels: Number of image channels (1 for grayscale, 3 for RGB)
        img_size: Height/width of the square image
    """
    # Calculate grid size based on number of images (assumes perfect square)
    grid_dim = int(np.sqrt(len(images)))
    rows, cols = grid_dim, grid_dim
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if im_channels == 1:
                # Grayscale: reshape to (H, W)
                img = np.reshape(images[i], (img_size, img_size))
            else:
                # RGB: reshape to (H, W, C)
                img = np.reshape(images[i], (img_size, img_size, im_channels))

            # Use nearest-neighbor interpolation for sharp pixel-perfect display
            ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved generated images to {save_path}")


def generate_images(model_path, cfg, num_images=25):
    """
    Generate images from a trained model.

    Args:
        model_path: Path to the trained model weights
        cfg: Configuration object
        num_images: Number of images to generate (default: 25 for 5x5 grid)

    Returns:
        List of generated images as numpy arrays
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from: {model_path}")

    # Initialize diffusion process
    diffusion = cfg.diffusion().to(device)

    # Initialize and load model
    model = Unet(
        im_channels=cfg.im_channels,
        down_ch=cfg.unet_down_ch,
        mid_ch=cfg.unet_mid_ch,
        up_ch=cfg.unet_up_ch,
        down_sample=cfg.unet_down_sample,
        t_emb_dim=cfg.unet_t_emb_dim,
        num_downc_layers=cfg.unet_num_downc_layers,
        num_midc_layers=cfg.unet_num_midc_layers,
        num_upc_layers=cfg.unet_num_upc_layers,
        dropout=cfg.unet_dropout,
    ).to(device)

    model.load_state_dict(torch.load(str(model_path), weights_only=True))
    model.eval()

    print(f"Generating {num_images} images...")
    generated_imgs = []

    with torch.no_grad():
        for i in tqdm(range(num_images), desc="Generating images"):
            img = diffusion.generate(cfg, model, return_intermediate=False)
            generated_imgs.append(img.flatten())

    return generated_imgs


if __name__ == "__main__":
    print("="*60)
    print("DDPM CIFAR-10 Image Generation (AdamW)")
    print("="*60)

    # Model path
    model_path = Path("outputs/ddpm_cifar10/AdamW/ddpm_unet_epoch_80.pth")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Available models:")
        model_dir = Path("outputs/ddpm_cifar10/AdamW")
        if model_dir.exists():
            for f in model_dir.glob("*.pth"):
                print(f"  - {f}")
        exit(1)

    # Generate 25 images (5x5 grid)
    num_images = 25
    generated_imgs = generate_images(model_path, CONFIG, num_images=num_images)

    # Save visualization with nearest interpolation
    output_path = Path("outputs/ddpm_cifar10/AdamW/generated_test_5x5_nearest.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_generated_images_nearest(
        generated_imgs,
        output_path,
        cmap=CONFIG.cmap,
        im_channels=CONFIG.im_channels,
        img_size=CONFIG.img_size
    )

    print("="*60)
    print("Generation completed!")
    print(f"Output saved to: {output_path}")
    print("="*60)
