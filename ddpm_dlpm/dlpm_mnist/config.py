from pathlib import Path
from ddpm_dlpm.data_mnist import CustomMnistDataset
from ddpm_dlpm.process import DLPM


class CONFIG:
    # Base directories
    data_dir = Path("data/mnist")
    model_dir = Path("outputs/dlpm_mnist")
    outputs_dir = Path("outputs/dlpm_mnist")

    # Dataset configuration
    train_data_path = data_dir / "train.csv"
    dataset_class = CustomMnistDataset

    # Diffusion process configuration
    diffusion = DLPM
    alpha = 1.7  # Tail index for alpha-stable distribution (1 < alpha <= 2)

    # Training parameters
    num_epochs = 150
    checkpoint_epochs = [2, 25, 50, 75, 100, 125, 150]
    num_timesteps = 1000
    batch_size = 128
    img_size = 28
    im_channels = 1
    num_img_to_generate = 64
    cmap = "gray"  # Colormap for visualization

    # U-Net architecture (4 levels, smaller model for MNIST)
    unet_down_ch = [32, 64, 128, 256]
    unet_mid_ch = [256, 256, 128]
    unet_up_ch = [256, 128, 64, 16]
    unet_down_sample = [True, True, False]
    unet_num_downc_layers = 2
    unet_num_midc_layers = 2
    unet_num_upc_layers = 2
    unet_t_emb_dim = 128
    unet_dropout = 0.0  # No dropout for MNIST

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
