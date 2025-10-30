from pathlib import Path
# Note: CustomCifar10Dataset needs to be created in ddpm/ddpm_cifar10/data.py
# from ddpm.ddpm_cifar10.data import CustomCifar10Dataset


class CONFIG:
    # Base directories
    data_dir = Path("data/cifar10")
    model_dir = Path("outputs/ddpm_cifar10")
    outputs_dir = Path("outputs/ddpm_cifar10")

    # Dataset configuration
    train_data_path = data_dir / "train"
    # dataset_class = CustomCifar10Dataset  # Uncomment after creating data.py

    # Training parameters
    num_epochs = 150
    checkpoint_epochs = [25, 50, 75, 100, 125, 150]
    num_timesteps = 1000
    batch_size = 128
    img_size = 32
    im_channels = 3
    num_img_to_generate = 64
    cmap = None  # No colormap for RGB images

    # Optimizer configurations
    optimizer_configs = {
        "Adam": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0
        }
    }