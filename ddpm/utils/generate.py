import torch

from ddpm.utils.process import BackwardProcess


def generate(cfg):
    """generate new images"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bp = BackwardProcess()

    model = torch.load(cfg.model_path).to(device)
    model.eval()

    x_t = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(x_t, torch.as_tensor(t).unsqueeze(0).to(device))
            x_t = bp.denoise(x_t, torch.as_tensor(t).to(device), noise_pred)

    x_t = torch.clamp(x_t, -1.0, 1.0).detach().cpu()
    x_t = (x_t + 1) / 2

    return x_t
