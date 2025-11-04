import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

def plot_cifar10_grid():
    """Plot 25 CIFAR-10 images in a 5x5 grid"""

    # Path to CIFAR-10 training images
    data_dir = Path("data/cifar10/train")

    # Get first 25 image paths
    image_paths = sorted(list(data_dir.glob("*.png")))[:25]

    if len(image_paths) < 25:
        print(f"Warning: Only found {len(image_paths)} images")
        return

    # Create 5x5 subplot grid
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle('CIFAR-10 Sample Images (5x5 Grid)', fontsize=16)

    # Plot each image
    for idx, (ax, img_path) in enumerate(zip(axes.flat, image_paths)):
        # Load and display image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        ax.imshow(img_array)
        ax.axis('off')
        ax.set_title(f'Image {idx+1}', fontsize=8)

    plt.tight_layout()
    plt.savefig('ddpm_dlpm/cifar10_samples_grid.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ddpm_dlpm/cifar10_samples_grid.png")
    plt.show()

if __name__ == "__main__":
    plot_cifar10_grid()
