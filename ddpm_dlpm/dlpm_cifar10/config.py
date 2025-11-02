from pathlib import Path
from ddpm_dlpm.data_cifar10 import CustomCifar10Dataset
from ddpm_dlpm.process import DLPM


class CONFIG:

    data_dir = Path("data/cifar10")
    model_dir = Path("outputs/dlpm_cifar10")
    outputs_dir = Path("outputs/dlpm_cifar10")

    train_data_path = data_dir / "train"
    dataset_class = CustomCifar10Dataset

    # Diffusion process configuration
    diffusion = DLPM
    alpha = 1.7  # Tail index for alpha-stable distribution (1 < alpha <= 2)

    num_epochs = 100
    checkpoint_epochs = [25, 50, 75, 100]
    num_timesteps = 1000
    batch_size = 128
    img_size = 32
    im_channels = 3
    num_img_to_generate = 16
    cmap = None

    # U-Net architecture (5 levels with 512 channels)
    unet_down_ch = [32, 64, 128, 256, 512]
    unet_mid_ch = [512, 512, 256]
    unet_up_ch = [512, 256, 128, 64, 32]
    unet_down_sample = [True, True, True, False]
    unet_num_downc_layers = 2
    unet_num_midc_layers = 2
    unet_num_upc_layers = 2
    unet_t_emb_dim = 128
    unet_dropout = 0.1  # Dropout for regularization (DDPM paper uses 0.1 for CIFAR10)

    # Optimizer configurations
    optimizer_configs = {
        "Adam": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0
        },
        "AdamW": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    }
