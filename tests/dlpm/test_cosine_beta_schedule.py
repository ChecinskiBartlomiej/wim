import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ddpm_dlpm_multigpu.process import DLPM
from ddpm_dlpm_multigpu.dlpm_cifar10.config import CONFIG as cfg


def test_cosine_beta_schedule():
    """
    Test the cosine beta schedule by plotting beta values over timesteps.

    The cosine schedule should:
    - Start small (near 0)
    - Gradually increase
    - End larger (near 1)
    - Be smooth (no sudden jumps)
    """

    # Create DLPM instance using config parameters
    dlpm = DLPM(num_time_steps=cfg.num_timesteps, alpha=cfg.alpha)

    # Get beta values
    betas = dlpm.betas.numpy()
    timesteps = np.arange(len(betas))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Cosine Beta Schedule Analysis (T={cfg.num_timesteps})', fontsize=14)

    # Calculate alphas and cumulative product
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)

    # Plot 1: Beta values over time
    axes[0].plot(timesteps, betas, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('Beta_t')
    axes[0].set_title('Beta Schedule')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative product (alpha_bar)
    axes[1].plot(timesteps, alphas_cumprod, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('Alpha_bar_t')
    axes[1].set_title('Cumulative Product of Alphas')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Statistics
    axes[2].axis('off')

    stats_text = f"""
    Beta Schedule Statistics:

    Min beta:     {betas.min():.6f}
    Max beta:     {betas.max():.6f}
    Mean beta:    {betas.mean():.6f}

    Beta[0]:      {betas[0]:.6f}
    Beta[500]:    {betas[500]:.6f}
    Beta[999]:    {betas[-1]:.6f}

    Alpha_bar[0]:    {alphas_cumprod[0]:.6f}
    Alpha_bar[500]:  {alphas_cumprod[500]:.6f}
    Alpha_bar[999]:  {alphas_cumprod[-1]:.8f}

    Expected behavior:
    - Betas should be in [0.001, 0.999]
    - Betas should increase smoothly
    - Alpha_bar should decrease to near 0
    """
    axes[2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                 family='monospace')

    plt.tight_layout()

    # Save plot
    output_dir = Path('tests_outputs/dlpm')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'test_cosine_beta_schedule.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print statistics
    print("\n" + "="*60)
    print("COSINE BETA SCHEDULE TEST")
    print("="*60)
    print(f"Number of timesteps: {cfg.num_timesteps}")
    print(f"Alpha parameter: {cfg.alpha}")
    print(f"\nBeta range: [{betas.min():.6f}, {betas.max():.6f}]")
    print(f"Mean beta: {betas.mean():.6f}")
    print(f"\nFirst 10 betas: {betas[:10]}")
    print(f"Last 10 betas: {betas[-10:]}")
    print("="*60)

    # Sanity checks
    assert betas.min() >= 0.001, "Beta values should be >= 0.001"
    assert betas.max() <= 0.999, "Beta values should be <= 0.999"
    assert np.all(np.diff(betas) >= 0), "Betas should be increasing"
    assert np.all(np.diff(alphas_cumprod) <= 0), "Alpha_bar should be decreasing"

    print("\n All sanity checks passed!")


if __name__ == "__main__":
    test_cosine_beta_schedule()
