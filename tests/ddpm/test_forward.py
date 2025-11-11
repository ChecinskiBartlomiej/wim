import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ddpm_dlpm.process import DDPM
from ddpm_dlpm.ddpm_cifar10.config import CONFIG as cfg


def test_forward_corruption():
    """
    Visualize forward diffusion process - image corruption over time.
    """

    # Create DDPM instance
    ddpm = DDPM(num_time_steps=cfg.num_timesteps)

    print("\n" + "="*60)
    print("FORWARD DIFFUSION VISUALIZATION (DDPM)")
    print("="*60)
    print(f"Number of timesteps: {cfg.num_timesteps}")
    print("="*60)

    # Load dataset and get one image
    print("\nLoading dataset...")
    dataset = cfg.dataset_class(
        cfg.train_data_path,
        use_horizontal_flip=False  # No augmentation for test
    )

    # Get first image
    x_0 = dataset[0]  # Returns image_tensor
    x_0 = x_0.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

    print(f"Image shape: {x_0.shape}")
    print(f"Image range: [{x_0.min():.3f}, {x_0.max():.3f}]")

    # Timesteps to visualize
    timesteps = [0, 1, 2, 5, 10, 50, 100, 250, 500, 999]

    # Generate noise once (same noise for all timesteps for fair comparison)
    noise = torch.randn_like(x_0)

    # Apply forward process at different timesteps
    print(f"\nApplying forward diffusion at timesteps: {timesteps}")
    corrupted_images = []

    for t_val in timesteps:
        if t_val == 0:
            x_t = x_0
        else:
            t = torch.tensor([t_val])
            x_t = ddpm.forward(x_0, noise, t)
        corrupted_images.append(x_t)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Forward Diffusion Process (DDPM - Gaussian Noise)', fontsize=16)

    axes = axes.flatten()

    for idx, (t_val, x_t) in enumerate(zip(timesteps, corrupted_images)):
        # Convert to image format using built-in function
        img = ddpm.tensor_to_image(x_t, cfg)[0]  # Returns list, get first image

        axes[idx].imshow(img)
        axes[idx].set_title(f't = {t_val}')
        axes[idx].axis('off')

    plt.tight_layout()

    # Save plot
    output_dir = Path('tests_outputs/ddpm')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'test_forward.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    print("\n Test completed!")


if __name__ == "__main__":
    test_forward_corruption()
