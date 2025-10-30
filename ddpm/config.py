from pathlib import Path


class CONFIG:
    # Base directories
    data_dir = Path("data/MNIST")
    model_dir = Path("outputs/ddpm_MNIST")
    outputs_dir = Path("outputs/ddpm_MNIST")

    # Training parameters
    num_epochs = 150
    checkpoint_epochs = [25, 50, 75, 100, 125, 150]
    num_timesteps = 1000
    batch_size = 128
    img_size = 28
    in_channels = 1
    num_img_to_generate = 64

    # Optimizer configurations
    optimizer_configs = {
        "Adam": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0
        },
        "SGD": {
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "nesterov": True
        },
        "AdamW": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    }
