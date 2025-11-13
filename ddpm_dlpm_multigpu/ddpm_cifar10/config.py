from pathlib import Path
from ddpm_dlpm_multigpu.custom_data import CustomCifar10Dataset
from ddpm_dlpm_multigpu.process import DDPM


class CONFIG:

    data_dir = Path("data/cifar10")
    model_dir = Path("outputs/ddpm_cifar10")
    outputs_dir = Path("outputs/ddpm_cifar10")
    train_data_path = data_dir / "train"

    diffusion = DDPM
    dataset_class = CustomCifar10Dataset

    use_horizontal_flip = True  

    num_epochs = 6000
    checkpoint_epochs = [100, 500, 1000, 1500, 2000, 2500, 3000, 4500, 6000]

    num_timesteps = 1000
    batch_size = 192
    img_size = 32
    im_channels = 3
    num_img_to_generate = 25

    cmap = None

    denoising_timestep_interval = 40  

    num_fid_images = 50000
    fid_batch_size = 128
    inception_path = Path("pretrained_models/inception_v3_imagenet.pth")
    
    unet_down_ch = [32, 64, 128, 256, 512]
    unet_mid_ch = [512, 512, 256]
    unet_up_ch = [512, 256, 128, 64, 32]
    unet_down_sample = [True, True, True, False]
    unet_num_downc_layers = 2
    unet_num_midc_layers = 2
    unet_num_upc_layers = 2
    unet_t_emb_dim = 128
    unet_dropout = 0.1  

    num_workers = 4 

    optimizer_configs = {
        "Adam": {
            "lr": 7e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0
        },
        "AdamW": {
            "lr": 7e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    }

    use_scheduler = False

    resume_checkpoint = None
    #resume_checkpoint = "outputs/ddpm_cifar10/AdamW/ddpm_unet_epoch_3000.pth"