"""Main script to train DLPM model on WIM supercomputer using Adam optimizer"""

from ddpm_dlpm_multigpu.dlpm_cifar10.config import CONFIG
from ddpm_dlpm_multigpu.training import train

if __name__ == "__main__":
    print("="*60)
    print("Starting DLPM training on CIFAR-10 with Adam optimizer")
    print("="*60)
    print(f"Epochs: {CONFIG.num_epochs}")
    print(f"Batch size: {CONFIG.batch_size}")
    print(f"Learning rate: {CONFIG.optimizer_configs['Adam']['lr']}")
    print(f"Alpha (tail index): {CONFIG.alpha}")
    print(f"Timesteps: {CONFIG.num_timesteps}")
    print(f"Image size: {CONFIG.img_size}x{CONFIG.img_size}")
    print(f"Channels: {CONFIG.im_channels} (RGB)")
    print(f"Checkpoint epochs: {CONFIG.checkpoint_epochs}")
    print("="*60 + "\n")

    train(CONFIG, optimizer_name="Adam")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
