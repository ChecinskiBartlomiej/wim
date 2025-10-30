"""
Visualization utilities for plotting training metrics and generated images.
General-purpose functions that work with any model/dataset.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_combined_stats(
    epoch_losses,
    all_grad_norms,
    all_zero_grad_pcts,
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
        all_zero_grad_pcts: List of zero gradient percentages per epoch
        all_weight_norms: List of weight norms per epoch
        all_zero_weight_pcts: List of zero weight percentages per epoch
        all_update_ratios: List of weight update ratios per epoch
        epochs: List of epoch numbers (e.g., [1, 2, 3, ...])
        save_path: Path to save the plot (should be Path object or string)
        title_suffix: Optional suffix for plot titles (e.g., optimizer name)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Loss, Grad Norm, Zero Grad %
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

    axes[0, 2].plot(epochs, all_zero_grad_pcts, 'r-o', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Zero Grad %', fontsize=12)
    axes[0, 2].set_title('Zero Gradient %', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

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

    axes[1, 2].plot(epochs, all_update_ratios, 'orange', linewidth=2, markersize=6, marker='o')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Update Ratio', fontsize=12)
    axes[1, 2].set_title('Weight Update Ratio', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_generated_images(images, save_path, grid_size=(8, 8), cmap, im_channels, img_size):
    """
    Plot a grid of generated images.

    Args:
        images: List of numpy arrays (flattened 1D images)
        save_path: Path to save the plot (should be Path object or string)
        grid_size: Tuple (rows, cols) for the grid layout
        cmap: Colormap for images ("gray" for grayscale, None for RGB)
        im_channels: Number of image channels (1 for grayscale, 3 for RGB)
        img_size: Height/width of the square image
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if im_channels == 1:
                # Grayscale: reshape to (H, W)
                img = np.reshape(images[i], (img_size, img_size))
            else:
                # RGB: reshape to (H, W, C)
                img = np.reshape(images[i], (img_size, img_size, im_channels))

            ax.imshow(img, cmap=cmap)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
