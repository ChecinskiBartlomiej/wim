import torch
from ddpm_dlpm.unet_utils import Unet

print("="*60)
print("YOUR MODEL ARCHITECTURES")
print("="*60)

# MNIST - smaller model (4 levels, no dropout)
model_mnist = Unet(
    im_channels=1,
    down_ch=[32, 64, 128, 256],
    mid_ch=[256, 256, 128],
    up_ch=[256, 128, 64, 16],
    down_sample=[True, True, False],
    dropout=0.0,
)
mnist_params = sum(p.numel() for p in model_mnist.parameters())
print(f"\nMNIST (4 levels, smaller):")
print(f"  down_ch = [32, 64, 128, 256]")
print(f"  dropout = 0.0")
print(f"  Parameters: {mnist_params:,} ({mnist_params/1e6:.2f}M)")

# CIFAR10 - larger model (5 levels with 512 + dropout)
model_cifar = Unet(
    im_channels=3,
    down_ch=[32, 64, 128, 256, 512],
    mid_ch=[512, 512, 256],
    up_ch=[512, 256, 128, 64, 32],
    down_sample=[True, True, True, False],
    dropout=0.1,
)
cifar_params = sum(p.numel() for p in model_cifar.parameters())
print(f"\nCIFAR10 (5 levels with 512):")
print(f"  down_ch = [32, 64, 128, 256, 512]")
print(f"  dropout = 0.1 (DDPM paper)")
print(f"  Parameters: {cifar_params:,} ({cifar_params/1e6:.2f}M)")

print("\n" + "="*60)
print("DDPM PAPER COMPARISON")
print("="*60)
print(f"CIFAR10 (32×32): 35.7M parameters")
print(f"  Your model: {cifar_params/35.7e6*100:.1f}% of DDPM size")
print(f"\nLSUN/CelebA-HQ (256×256): 114M parameters")
print(f"Large LSUN Bedroom: ~256M parameters")
