from pathlib import Path
from ddpm_dlpm.custom_data import CustomCifar10Dataset
from ddpm_dlpm.process import DDPM


class CONFIG:

    data_dir = Path("data/cifar10")
    model_dir = Path("outputs/ddpm_cifar10")
    outputs_dir = Path("outputs/ddpm_cifar10")


    train_data_path = data_dir / "train"
    dataset_class = CustomCifar10Dataset
    use_horizontal_flip = True  # DDPM paper: improves sample quality slightly for CIFAR10

    # Diffusion process configuration
    diffusion = DDPM

    num_epochs = 750
    checkpoint_epochs = [75, 150, 225, 300, 375, 450, 525, 600, 675, 750]
    num_timesteps = 1000
    batch_size = 192
    num_workers = 7  # Number of DataLoader workers
    img_size = 32
    im_channels = 3
    num_img_to_generate = 25  # Changed to 25 for 5x5 grid
    cmap = None

    # Denoising progress visualization
    denoising_timestep_interval = 40  # Save image every N timesteps (1000/40 = 25 timesteps)

    # FID calculation settings
    num_fid_images = 1280  # Number of images for FID calculation (5000 for testing, 50000 for final)
    inception_path = Path("pretrained_models/inception_v3_imagenet.pth")
    fid_batch_size = 64

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
            "lr": 2e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0
        },
        "AdamW": {
            "lr": 2e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    }