import torch
from pathlib import Path

from ddpm_dlpm.unet_utils import Unet
from ddpm_dlpm.process import DDPM, DLPM


def generate(cfg, model, diffusion):
    """Generate new images

    Args:
        cfg: Configuration object
        model: Either:
            - str/Path: Load model weights from this file path
            - nn.Module: Use this model directly (already loaded)
        diffusion: Diffusion process instance (DDPM or DLPM)
    """

    if isinstance(model, (str, Path)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model}")
        print(f"Device: {device}")
        unet = Unet(
            im_channels=cfg.im_channels,
            down_ch=cfg.unet_down_ch,
            mid_ch=cfg.unet_mid_ch,
            up_ch=cfg.unet_up_ch,
            down_sample=cfg.unet_down_sample,
            t_emb_dim=cfg.unet_t_emb_dim,
            num_downc_layers=cfg.unet_num_downc_layers,
            num_midc_layers=cfg.unet_num_midc_layers,
            num_upc_layers=cfg.unet_num_upc_layers,
            dropout=cfg.unet_dropout,
        ).to(device)
        unet.load_state_dict(torch.load(str(model), weights_only=True))
        unet.eval()
        model = unet
    else:
        device = next(model.parameters()).device

    x_t = torch.randn(1, cfg.im_channels, cfg.img_size, cfg.img_size).to(device)

    # Sample alpha-stable noise for DLPM
    if cfg.mode == "DLPM":
        a_samples = diffusion.sample_alpha_stable(size=(1, cfg.num_timesteps), device=device)

    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(x_t, torch.as_tensor(t).unsqueeze(0).to(device))

            if cfg.mode == "DDPM":
                x_t = diffusion.backward(x_t, t, noise_pred)
            else:  # DLPM
                x_t = diffusion.backward(x_t, t, noise_pred, a_samples)

    x_t = torch.clamp(x_t, -1.0, 1.0).detach().cpu()
    x_t = (x_t + 1) / 2

    return x_t
