from pathlib import Path
from ddpm_dlpm_multigpu.custom_data import CustomMnistDataset
from ddpm_dlpm_multigpu.process import DLPM


class CONFIG:

    data_dir = Path("data/mnist")
    model_dir = Path("outputs/dlpm_mnist")
    outputs_dir = Path("outputs/dlpm_mnist")

    diffusion = DLPM
    dataset_class = CustomMnistDataset

    use_horizontal_flip = False

    num_epochs = 3000
    checkpoint_epochs = [3000]

    train_data_path = data_dir / "train.csv"

    alpha = 1.7  
  
    num_timesteps = 1000
    batch_size = 192
    img_size = 28
    im_channels = 1
    num_img_to_generate = 25  

    cmap = "gray"  

    denoising_timestep_interval = 40  

    num_fid_images = 10000  
    fid_batch_size = 128
    inception_path = Path("pretrained_models/inception_v3_imagenet.pth")

    unet_down_ch = [32, 64, 128, 256]
    unet_mid_ch = [256, 256, 128]
    unet_up_ch = [256, 128, 64, 16]
    unet_down_sample = [True, True, False]
    unet_num_downc_layers = 2
    unet_num_midc_layers = 2
    unet_num_upc_layers = 2
    unet_t_emb_dim = 128
    unet_dropout = 0.0  # No dropout for MNIST
    net_attention_resolutions = [14] 

    num_workers = 4

    optimizer_configs = {
        "Adam": {
            "lr": 6e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0
        },
        "AdamW": {
            "lr": 6e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    }

    use_scheduler = False

    resume_checkpoint = None
    #resume_checkpoint = "outputs/ddpm_cifar10/AdamW/ddpm_unet_epoch_3000.pth"

    eta = 0