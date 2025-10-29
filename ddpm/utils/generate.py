import torch

from ddpm.utils.process import BackwardProcess
from ddpm.utils.unet_utils import Unet


def generate(cfg):
    """generate new images"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bp = BackwardProcess()

    # Move BackwardProcess tensors to device
    bp.betas = bp.betas.to(device)
    bp.sqrt_betas = bp.sqrt_betas.to(device)
    bp.alphas = bp.alphas.to(device)
    bp.sqrt_alphas = bp.sqrt_alphas.to(device)
    bp.alpha_bars = bp.alpha_bars.to(device)
    bp.sqrt_alpha_bars = bp.sqrt_alpha_bars.to(device)
    bp.sqrt_one_minus_alpha_bars = bp.sqrt_one_minus_alpha_bars.to(device)

    # Create model architecture
    model = Unet().to(device)

    # Load only the weights (safe, no arbitrary code execution)
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True))
    model.eval()

    x_t = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(x_t, torch.as_tensor(t).unsqueeze(0).to(device))
            x_t = bp.denoise(x_t, torch.as_tensor(t).to(device), noise_pred)

    x_t = torch.clamp(x_t, -1.0, 1.0).detach().cpu()
    x_t = (x_t + 1) / 2

    return x_t
