"""Main script to train DDPM model on WIM supercomputer using SGD optimizer"""

from ddpm.config import CONFIG
from ddpm.training import train

if __name__ == "__main__":
    print("="*60)
    print("Starting DDPM training with SGD optimizer")
    print("="*60)
    print(f"Epochs: {CONFIG.num_epochs}")
    print(f"Batch size: {CONFIG.batch_size}")
    print(f"Learning rate: {CONFIG.optimizer_configs['SGD']['lr']}")
    print(f"Momentum: {CONFIG.optimizer_configs['SGD']['momentum']}")
    print(f"Nesterov: {CONFIG.optimizer_configs['SGD']['nesterov']}")
    print(f"Timesteps: {CONFIG.num_timesteps}")
    print(f"Checkpoint epochs: {CONFIG.checkpoint_epochs}")
    print("="*60 + "\n")

    train(CONFIG, optimizer_name="SGD")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)