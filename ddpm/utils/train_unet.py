"""
Main script to train DDPM model on WIM supercomputer
"""

from ddpm.utils.config import CONFIG
from ddpm.utils.training import train

if __name__ == "__main__":
    print("Starting DDPM training...")
    print(f"Epochs: {CONFIG.num_epochs}")
    print(f"Batch size: {CONFIG.batch_size}")
    print(f"Learning rate: {CONFIG.lr}")
    print(f"Timesteps: {CONFIG.num_timesteps}")

    train(CONFIG)

    print("Training completed!")
