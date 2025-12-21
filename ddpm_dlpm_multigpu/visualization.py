import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_combined_stats(
    epoch_losses,
    all_grad_norms,
    all_grad_below_thresh,
    all_weight_norms,
    all_zero_weight_pcts,
    all_update_ratios,
    epochs,
    save_path,
    title_suffix=""
):
    """
    Plot combined 2x3 grid of training statistics.

    Args:
        epoch_losses: List of loss values per epoch
        all_grad_norms: List of gradient norms per epoch
        all_grad_below_thresh: Dict mapping threshold to list of percentages per epoch
                              e.g., {1e-10: [10.2, 11.5, ...], 1e-8: [45.3, 46.1, ...], ...}
        all_weight_norms: List of weight norms per epoch
        all_zero_weight_pcts: List of zero weight percentages per epoch
        all_update_ratios: List of weight update ratios per epoch
        epochs: List of epoch numbers (e.g., [1, 2, 3, ...])
        save_path: Path to save the plot (should be Path object or string)
        title_suffix: Optional suffix for plot titles (e.g., optimizer name)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Loss, Grad Norm, Multi-threshold Grad %
    axes[0, 0].plot(epochs, epoch_losses, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title(f'Training Loss{title_suffix}', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, all_grad_norms, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Gradient Norm', fontsize=12)
    axes[0, 1].set_title('Gradient Norm', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Plot multiple threshold curves with highly distinct colors
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']  # Red, Blue, Green, Orange, Purple, Turquoise
    sorted_thresholds = sorted(all_grad_below_thresh.keys(), reverse=True)  # Largest to smallest

    for i, thresh in enumerate(sorted_thresholds):
        color = colors[i % len(colors)]
        label = f'< {thresh:.0e}'
        axes[0, 2].plot(epochs, all_grad_below_thresh[thresh],
                       color=color, linewidth=2, markersize=5, marker='o', label=label)

    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Gradient %', fontsize=12)
    axes[0, 2].set_title('Gradients Below Threshold (%)', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(loc='best', fontsize=10, framealpha=0.9)

    # Row 2: Weight Norm, Zero Weight %, Update Ratio
    axes[1, 0].plot(epochs, all_weight_norms, 'c-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Weight Norm', fontsize=12)
    axes[1, 0].set_title('Weight Norm', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    axes[1, 1].plot(epochs, all_zero_weight_pcts, 'm-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Zero Weight %', fontsize=12)
    axes[1, 1].set_title('Zero Weight %', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Skip first epoch for update ratio to avoid scale distortion
    if len(epochs) > 1:
        axes[1, 2].plot(epochs[1:], all_update_ratios[1:], 'orange', linewidth=2, markersize=6, marker='o')
    else:
        axes[1, 2].plot(epochs, all_update_ratios, 'orange', linewidth=2, markersize=6, marker='o')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Update Ratio', fontsize=12)
    axes[1, 2].set_title('Weight Update Ratio (excl. epoch 1)', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_generated_images(images, save_path, cmap, im_channels, img_size):
    """
    Plot a grid of generated images.

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

            ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_bucket_losses(all_bucket_losses, epochs, num_timesteps, save_path, title_suffix=""):
    """
    Plot loss per timestep bucket over epochs.

    Args:
        all_bucket_losses: List of lists, shape (num_epochs, 10), where each inner list
                          contains mean loss for each of the 10 timestep buckets
        epochs: List of epoch numbers (e.g., [1, 2, 3, ...])
        num_timesteps: Total number of timesteps (e.g., 1000) for labeling
        save_path: Path to save the plot
        title_suffix: Optional suffix for plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    num_buckets = len(all_bucket_losses[0]) if all_bucket_losses else 10
    bucket_size = num_timesteps // num_buckets

    # Use a colormap from blue (t=0, clean) to red (t=max, noisy)
    cmap_colors = plt.cm.coolwarm(np.linspace(0, 1, num_buckets))

    # Convert to numpy for easier slicing
    bucket_losses_array = np.array(all_bucket_losses)  # shape: (num_epochs, 10)

    for bucket_idx in range(num_buckets):
        start_t = bucket_idx * bucket_size
        end_t = (bucket_idx + 1) * bucket_size - 1
        label = f't={start_t}-{end_t}'
        ax.plot(epochs, bucket_losses_array[:, bucket_idx],
                color=cmap_colors[bucket_idx], linewidth=2, label=label)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Loss by Timestep Bucket{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_denoising_progress(all_intermediate_images, all_timesteps, save_path, cmap, im_channels, img_size):
    """
    Plot the denoising process for multiple generated images.
    Each row shows one generated image, each column shows a timestep in the denoising process.

    Args:
        all_intermediate_images: List of lists, where each inner list contains
                                intermediate images for one generation
                                [[img1_t999, img1_t950, ..., img1_t0], [img2_t999, ...], ...]
        all_timesteps: List of timestep values (same for all images)
        save_path: Path to save the plot (should be Path object or string)
        cmap: Colormap for images ("gray" for grayscale, None for RGB)
        im_channels: Number of image channels (1 for grayscale, 3 for RGB)
        img_size: Height/width of the square image
    """
    num_images = len(all_intermediate_images)
    num_timesteps = len(all_timesteps)

    # Calculate appropriate figure size based on number of images and timesteps
    fig_width = min(num_timesteps * 1.5, 30)  # Cap at 30 inches
    fig_height = min(num_images * 1.5, 30)    # Cap at 30 inches

    fig, axes = plt.subplots(num_images, num_timesteps, figsize=(fig_width, fig_height))

    # Plot images
    for row_idx in range(num_images):
        intermediate_images = all_intermediate_images[row_idx]

        for col_idx in range(num_timesteps):
            ax = axes[row_idx, col_idx]

            if col_idx < len(intermediate_images):
                img = intermediate_images[col_idx]

                ax.imshow(img, cmap=cmap, interpolation='nearest')

                # Add timestep label on top row
                if row_idx == 0:
                    timestep = all_timesteps[col_idx]
                    ax.set_title(f't={timestep}', fontsize=8, fontweight='bold')

                # Add image number label on first column
                if col_idx == 0:
                    ax.set_ylabel(f'#{row_idx + 1}', fontsize=8, fontweight='bold', rotation=0, labelpad=15)

            ax.axis('off')

    plt.suptitle('Denoising Progress: From Noise to Generated Images', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()
