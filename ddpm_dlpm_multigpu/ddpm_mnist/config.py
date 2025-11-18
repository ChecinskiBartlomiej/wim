from pathlib import Path
from ddpm_dlpm_multigpu.custom_data import CustomMnistDataset
from ddpm_dlpm_multigpu.process import DDPM


class CONFIG:
    # Base directories
    data_dir = Path("data/mnist")
    model_dir = Path("outputs/ddpm_mnist")
    outputs_dir = Path("outputs/ddpm_mnist")

    # Dataset configuration
    train_data_path = data_dir / "train.csv"
    dataset_class = CustomMnistDataset
    use_horizontal_flip = False  # No horizontal flips for MNIST

    # Diffusion process configuration
    diffusion = DDPM  
 
    # Training parameters
    num_epochs = 3000
    checkpoint_epochs = [3000]
    num_timesteps = 1000
    batch_size = 192
    num_workers = 4  # Number of DataLoader workers per GPU
    img_size = 28
    im_channels = 1
    num_img_to_generate = 25  # Changed to 25 for 5x5 grid
    cmap = "gray"  # Colormap for visualization

    # Denoising progress visualization
    denoising_timestep_interval = 40  # Save image every N timesteps (1000/40 = 25 timesteps)

    # FID calculation settings
    num_fid_images = 10000  # Number of images for FID calculation (5000 for testing, 50000 for final)
    inception_path = Path("pretrained_models/inception_v3_imagenet.pth")
    fid_batch_size = 64

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
    unet_attention_resolutions = [14] 

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
