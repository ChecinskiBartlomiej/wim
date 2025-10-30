import torch

from ddpm.process import BackwardProcess
from ddpm.unet_utils import Unet


def generate(cfg, model=None):
    """generate new images

    Args:
        cfg: Configuration object
        model: Optional pre-trained model. If None, loads from disk.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bp = BackwardProcess()

    # Move BackwardProcess tensors to device
    bp.betas = bp.betas.to(device)
    bp.sqrt_betas = bp.sqrt_betas.to(device)
    bp.alphas = bp.alphas.to(device)
    bp.sqrt_alphas = bp.sqrt_alphas.to(device)
    bp.alpha_bars = bp.alpha_bars.to(device)
    bp.sqrt_alpha_bars = bp.sqrt_alpha_bars.to(device)
    bp.sqrt_one_minus_alpha_bars = bp.sqrt_one_minus_alpha_bars.to(device)

    # Use provided model or load from disk
    if model is None:
        print(f"Device: {device}")
        model = Unet().to(device)
        model_path = cfg.model_dir / "ddpm_unet.pth"
        model.load_state_dict(torch.load(str(model_path), weights_only=True))
        model.eval()
    else:
        # Use the provided model (already on device and in eval mode)
        device = next(model.parameters()).device

    x_t = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(x_t, torch.as_tensor(t).unsqueeze(0).to(device))
            x_t = bp.denoise(x_t, torch.as_tensor(t).to(device), noise_pred)

    x_t = torch.clamp(x_t, -1.0, 1.0).detach().cpu()
    x_t = (x_t + 1) / 2

    return x_t
