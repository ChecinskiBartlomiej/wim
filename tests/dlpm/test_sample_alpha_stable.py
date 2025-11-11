import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from scipy.stats import levy_stable
from ddpm_dlpm_multigpu.process import DLPM
from ddpm_dlpm_multigpu.dlpm_cifar10.config import CONFIG as cfg

def plot_histogram_with_pdf(ax, samples, alpha_half, c_A, title):
    """
    Plot histogram with theoretical PDF overlay.
    """
    # Clip only right side extreme values
    x_min = 0
    x_max = np.percentile(samples, 99)

    # Create histogram
    ax.hist(samples, bins=100, range=(x_min, x_max), density=True,
            alpha=0.7, color='blue', edgecolor='black')

    # Add theoretical PDF
    x_range = np.linspace(x_min, x_max, 500)
    pdf_values = levy_stable.pdf(x_range, alpha_half, beta=1, loc=0, scale=c_A)
    ax.plot(x_range, pdf_values, 'r-', linewidth=2, label='Theoretical PDF')

    # Set labels and formatting
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def time_sampling(dlpm, size, description, loop=False):
    """
    Time the sampling process and return samples and elapsed time.

    Args:
        dlpm: DLPM instance
        size: size argument for sample_alpha_stable
        description: description to print
        loop: if True, call sample_alpha_stable size times with size=1
              if False, call sample_alpha_stable once with given size

    Returns:
        (samples, elapsed_time) where samples is numpy array
    """
    print(f"\n{description}")
    start_time = time.time()

    if loop:
        samples_list = []
        for _ in range(size):
            sample = dlpm.sample_alpha_stable(size=1)
            samples_list.append(sample.item())
        samples = np.array(samples_list)
    else:
        samples = dlpm.sample_alpha_stable(size=size)
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()

    elapsed_time = time.time() - start_time
    print(f"   Time: {elapsed_time:.4f} seconds")
    print(f"   Shape: {samples.shape}")

    return samples, elapsed_time


def test_sample_alpha_stable():
    """
    Test alpha-stable sampling by checking sizes and visualizing distributions.
    """

    # Create DLPM instance
    dlpm = DLPM(num_time_steps=cfg.num_timesteps, alpha=cfg.alpha)

    print("\n" + "="*60)
    print("ALPHA-STABLE SAMPLING TEST")
    print("="*60)
    print(f"Alpha parameter: {cfg.alpha}")
    print(f"Alpha/2: {cfg.alpha / 2.0}")
    print("="*60)

    # Test 1: Size assertions
    print("\n1. Testing size assertions...")

    # Test scalar size
    samples_scalar = dlpm.sample_alpha_stable(size=100)
    assert samples_scalar.shape == (100,), f"Expected shape (100,), got {samples_scalar.shape}"
    print(f"    Scalar size: {samples_scalar.shape}")

    # Test tuple size
    samples_2d = dlpm.sample_alpha_stable(size=(10, 20))
    assert samples_2d.shape == (10, 20), f"Expected shape (10, 20), got {samples_2d.shape}"
    print(f"    Tuple size (10, 20): {samples_2d.shape}")

    # Test 3D size
    samples_3d = dlpm.sample_alpha_stable(size=(5, 10, 15))
    assert samples_3d.shape == (5, 10, 15), f"Expected shape (5, 10, 15), got {samples_3d.shape}"
    print(f"    Tuple size (5, 10, 15): {samples_3d.shape}")

    # Test 2: Sample num_samples times with size=1
    num_samples = 300000
    samples_individual, time_individual = time_sampling(
        dlpm, num_samples, f"2. Sampling {num_samples} times with size=1...", loop=True)

    # Test 3: Sample once with size=num_samples
    samples_batch, time_batch = time_sampling(
        dlpm, num_samples, f"3. Sampling once with size={num_samples}...")

    # Test 4: Sample once with size=(3, num_samples)
    samples_batch_2, time_batch_2 = time_sampling(
        dlpm, (3, num_samples), f"4. Sampling once with size=(3, {num_samples})...")

    # Calculate theoretical PDF parameters (same as in DLPM.sample_alpha_stable)
    alpha_half = cfg.alpha / 2.0
    c_A = np.cos(np.pi * cfg.alpha / 4.0) ** (2.0 / cfg.alpha)

    # Create visualization - 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Alpha-Stable Distribution Sampling (alpha={cfg.alpha})\nX-axis clipped at 99th percentile', fontsize=14)

    # Row 1, Col 1: num_samples individual samples (size=1 called num_samples times)
    plot_histogram_with_pdf(axes[0, 0], samples_individual, alpha_half, c_A,
                            f'{num_samples} x sample(size=1)')

    # Row 1, Col 2: Batch sampling (size=num_samples called once)
    plot_histogram_with_pdf(axes[0, 1], samples_batch, alpha_half, c_A,
                            f'1 x sample(size={num_samples})')

    # Row 1, Col 3: Hide (not needed)
    axes[0, 2].axis('off')

    # Row 2, Cols 1-3: Three samples from (3, num_samples)
    for i in range(3):
        samples_row = samples_batch_2[i]
        plot_histogram_with_pdf(axes[1, i], samples_row, alpha_half, c_A,
                                f'sample(size=(3, {num_samples}))[{i}]')

    plt.tight_layout()

    # Save plot
    output_dir = Path('tests_outputs/dlpm')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'test_sample_alpha_stable.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Final statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"All shape assertions: PASSED")
    print(f"Individual sampling ({num_samples} x size=1):    {time_individual:.4f}s")
    print(f"Batch sampling (size={num_samples}):             {time_batch:.4f}s")
    print(f"Batch sampling (size=(3, {num_samples})):        {time_batch_2:.4f}s")
    print("="*60)

    print("\n All tests passed!")


if __name__ == "__main__":
    test_sample_alpha_stable()
