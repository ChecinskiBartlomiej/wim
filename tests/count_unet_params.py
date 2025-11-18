import torch
from ddpm_dlpm_multigpu.unet import Unet
from ddpm_dlpm_multigpu.metrics import count_parameters


print("=" * 80)
print("U-NET PARAMETER COMPARISON")
print("=" * 80)

print("\n1. MNIST U-Net (4 levels - smaller model)")
print("-" * 80)

model_mnist = Unet(
    im_channels=1,
    down_ch=[32, 64, 128, 256],
    mid_ch=[256, 256, 128],
    up_ch=[256, 128, 64, 16],
    down_sample=[True, True, False],
    t_emb_dim=128,
    num_downc_layers=2,
    num_midc_layers=2,
    num_upc_layers=2,
    dropout=0.0,
    img_size=28,
    attention_resolutions=[16],
)

params_mnist = count_parameters(model_mnist)
print(f"Architecture:")
print(f"  Image channels: 1")
print(f"  Down channels: [32, 64, 128, 256]")
print(f"  Mid channels:  [256, 256, 128]")
print(f"  Up channels:   [256, 128, 64, 16]")
print(f"  Dropout: 0.0")
print(f"\nParameters:")
print(f"  Total:     {params_mnist['total_params']:,}")
print(f"  Trainable: {params_mnist['trainable_params']:,}")
print(f"  Size:      {params_mnist['total_params'] / 1e6:.2f}M")

print("\n" + "=" * 80)
print("2. CIFAR10 U-Net (5 levels with 512 channels max)")
print("-" * 80)

model_cifar = Unet(
    im_channels=3,
    down_ch=[32, 64, 128, 256, 512],
    mid_ch=[512, 512, 256],
    up_ch=[512, 256, 128, 64, 32],
    down_sample=[True, True, True, False],
    t_emb_dim=128,
    num_downc_layers=2,
    num_midc_layers=2,
    num_upc_layers=2,
    dropout=0.1,
    img_size=32,
    attention_resolutions=[16],
)

params_cifar = count_parameters(model_cifar)
print(f"Architecture:")
print(f"  Image channels: 3")
print(f"  Down channels: [32, 64, 128, 256, 512]")
print(f"  Mid channels:  [512, 512, 256]")
print(f"  Up channels:   [512, 256, 128, 64, 32]")
print(f"  Dropout: 0.1")
print(f"\nParameters:")
print(f"  Total:     {params_cifar['total_params']:,}")
print(f"  Trainable: {params_cifar['trainable_params']:,}")
print(f"  Size:      {params_cifar['total_params'] / 1e6:.2f}M")

print("\n" + "=" * 80)
print("3. Large U-Net for 64x64 images, (7 levels with 512 channels max)")
print("-" * 80)

model_large = Unet(
    im_channels=3,
    down_ch=[32, 64, 128, 256, 256, 512, 512],
    mid_ch=[512, 512, 256],
    up_ch=[512, 512, 256, 256, 128, 64, 32],
    down_sample=[True, True, True, True, True, False],
    t_emb_dim=128,
    num_downc_layers=2,
    num_midc_layers=2,
    num_upc_layers=2,
    dropout=0.1,
    img_size=32,
    attention_resolutions=[16],
)

params_large = count_parameters(model_large)
print(f"Architecture:")
print(f"  Image channels: 3")
print(f"  Down channels: [32, 64, 128, 256, 256, 512, 512]")
print(f"  Mid channels:  [512, 512, 256]")
print(f"  Up channels:   [512, 512, 256, 256, 128, 64, 32]")
print(f"  Dropout: 0.1")
print(f"  Attention: [16] (16×16 only)")
print(f"\nParameters:")
print(f"  Total:     {params_large['total_params']:,}")
print(f"  Trainable: {params_large['trainable_params']:,}")
print(f"  Size:      {params_large['total_params'] / 1e6:.2f}M")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

mnist_size = params_mnist['total_params'] / 1e6
cifar_size = params_cifar['total_params'] / 1e6
large_size = params_large['total_params'] / 1e6

print(f"\n{'Model':<20} {'Levels':<10} {'Max Channels':<15} {'Parameters':<20} {'Size (MB)':<15}")
print("-" * 80)
print(f"{'MNIST':<20} {'4':<10} {'256':<15} {params_mnist['total_params']:>19,} {mnist_size:>14.2f}M")
print(f"{'CIFAR10':<20} {'5':<10} {'512':<15} {params_cifar['total_params']:>19,} {cifar_size:>14.2f}M")
print(f"{'Large':<20} {'7':<10} {'512':<15} {params_large['total_params']:>19,} {large_size:>14.2f}M")

print(f"\nRelative to MNIST:")
print(f"  CIFAR10: {cifar_size / mnist_size:.2f}x larger")
print(f"  Large:   {large_size / mnist_size:.2f}x larger")

print(f"\nRelative to DDPM paper (CIFAR10 = 35.7M):")
print(f"  Our CIFAR10: {cifar_size / 35.7 * 100:.1f}% of DDPM size")
print(f"  Our Large:   {large_size / 35.7 * 100:.1f}% of DDPM size")

print("\n" + "=" * 80)
