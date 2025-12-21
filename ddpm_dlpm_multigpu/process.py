import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import levy_stable
from pathlib import Path
from ddpm_dlpm_multigpu.unet import Unet


class DiffusionProcess(ABC):
    """Abstract base class for diffusion processes (DDPM, DLPM)"""

    @abstractmethod
    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        """Initialize diffusion process parameters"""
        pass

    @abstractmethod
    def forward(self, x_0, noise, t):
        """Forward diffusion process - add noise to clean images"""
        pass

    @abstractmethod
    def backward(self, x_t, t, noise_prediction):
        """Backward diffusion process - denoise images"""
        pass

    def backward_ddim(self, x_t, t, noise_prediction):
        """Backward diffusion process - denoise images using DDIM strategy"""
        pass

    @abstractmethod
    def to(self, device):
        """Move parameters to device"""
        pass

    @abstractmethod
    def get_name(self):
        """Return the name of the diffusion process"""
        pass

    @abstractmethod
    def get_noise(self, imgs, device):
        """Generate noise samples for training target"""
        pass

    @abstractmethod
    def get_loss(self):
        """Return loss function"""
        pass

    @staticmethod
    def tensor_to_image(x_t, cfg):
        """
        Convert tensor in [-1, 1] to numpy images in [0, 255] uint8.

        Args:
            x_t: Tensor [B, C, H, W] in range [-1, 1]
            cfg: Configuration object

        Returns:
            list: List of numpy arrays, each in format (H, W, C) for RGB or (H, W) for grayscale,
                  scaled to [0, 255] as uint8. Length = B.
        """
        
        x_t = torch.clamp(x_t, -1.0, 1.0).detach().cpu()
        x_t = (x_t + 1) / 2

        batch_size = x_t.shape[0]
        images = []

        for i in range(batch_size):
            img = x_t[i].numpy()
            img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]

            if cfg.im_channels == 1:
                img = img.squeeze()

            img = 255 * img
            images.append(img.astype(np.uint8))

        return images

    def generate_samples(self, cfg, model, return_intermediate=False, intermediate_step=40, batch_size=1):
        """
        Generate samples using the diffusion process (backward denoising).

        Args:
            cfg: Configuration object
            model: Either:
                - str/Path: Load model weights from this file path
                - nn.Module: Use this model directly (already loaded)
            return_intermediate: If True, return samples at intermediate timesteps
            intermediate_step: Step size for saving intermediate samples (e.g., every 40 timesteps)
            batch_size: Number of images to generate in parallel (default: 1)

        Returns:
            If return_intermediate=False:
                Tensor: Generated samples [B, C, H, W] in range [-1, 1]
            If return_intermediate=True:
                tuple: (intermediate_samples, timesteps)
                    - intermediate_samples: List of tensors [B, C, H, W] in range [-1, 1]
                    - timesteps: List of corresponding timestep values
        """
        # Handle model loading
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
                img_size=cfg.img_size,
                attention_resolutions=cfg.unet_attention_resolutions,
            ).to(device)
            unet.load_state_dict(torch.load(str(model), weights_only=True))
            unet.eval()
            model = unet
        else:
            device = next(model.parameters()).device

        # Sample noise for DDPM
        if self.get_name() == "DDPM":
            x_t = torch.randn(batch_size, cfg.im_channels, cfg.img_size, cfg.img_size).to(device)

        # Sample noise for DLPM and precompute sigma_sq_all
        if self.get_name() == "DLPM":
            clamp_a = getattr(cfg, 'clamp_a', None)
            a_samples = self.sample_alpha_stable(size=(batch_size, cfg.num_timesteps), device=device, clamp_a=clamp_a)
            x_t = self.sigma_bars[-1] * torch.sqrt(self.sample_alpha_stable(size=(batch_size, 1, 1, 1), device=device, clamp_a=clamp_a)) * torch.randn(batch_size, cfg.im_channels, cfg.img_size, cfg.img_size, device=device)
            sigma_sq_all = self._precompute_all_sigma_sq(a_samples, device)

        intermediate_samples = []
        timesteps = []

        # Backward diffusion process
        with torch.no_grad():
            for t in reversed(range(cfg.num_timesteps)):
                noise_pred = model(x_t, torch.as_tensor(t).unsqueeze(0).repeat(batch_size).to(device))

                if self.get_name() == "DDPM":
                    x_t = self.backward(x_t, t, noise_pred)
                else:  # DLPM
                    x_t = self.backward(x_t, t, noise_pred, sigma_sq_all)

                if return_intermediate and t % intermediate_step == 0:
                    intermediate_samples.append(x_t)
                    timesteps.append(t)

        if not return_intermediate:
            return x_t  

        return intermediate_samples, timesteps

    def generate(self, cfg, model, return_intermediate=False, intermediate_step=40, batch_size=1):
        """
        Generate new images using the diffusion process.

        This is a convenience wrapper around generate_samples() that converts
        tensors to numpy uint8 images.

        Args:
            cfg: Configuration object
            model: Either:
                - str/Path: Load model weights from this file path
                - nn.Module: Use this model directly (already loaded)
            return_intermediate: If True, return images at intermediate timesteps
            intermediate_step: Step size for saving intermediate images (default: 1)
            batch_size: Number of images to generate in parallel (default: 1)

        Returns:
            If return_intermediate=False:
                list: List of numpy arrays, each in format (H, W, C) or (H, W) for grayscale,
                      scaled to [0, 255] as uint8. Length = batch_size.
            If return_intermediate=True:
                tuple: (all_intermediate_images, timesteps)
                    - all_intermediate_images: List of lists, where each inner list contains
                      intermediate images for one sample. Length = batch_size.
                    - timesteps: List of corresponding timestep values
        """
        # Generate samples (tensors in [-1, 1])
        if return_intermediate:
            intermediate_samples, timesteps = self.generate_samples(cfg, model, return_intermediate=True, intermediate_step=intermediate_step, batch_size=batch_size)

            # intermediate_samples is a list of tensors [B, C, H, W]
            # Reorganize to batch_size lists of intermediate images
            all_intermediate_images = [[] for _ in range(batch_size)]

            for sample_tensor in intermediate_samples:
                images = self.tensor_to_image(sample_tensor, cfg)
                for i in range(batch_size):
                    all_intermediate_images[i].append(images[i])

            return all_intermediate_images, timesteps

        else:
            samples = self.generate_samples(cfg, model, return_intermediate=False, batch_size=batch_size)

            return self.tensor_to_image(samples, cfg)


class DDPM(DiffusionProcess):
    """Denoising Diffusion Probabilistic Model (DDPM) implementation"""

    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initialize DDPM parameters using linear beta schedule.

        Args:
            num_time_steps: Number of diffusion steps
            beta_start: Starting value of beta schedule
            beta_end: Ending value of beta schedule
        """
        self.num_time_steps = num_time_steps
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.sqrt_betas = torch.sqrt(self.betas)
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def forward(self, x_0, noise, t):
        """
        Args:
            x_0: Clean images [batch_size, channels, height, width]
            noise: Random noise [batch_size, channels, height, width]
            t: Timestep indices [batch_size]

        Returns:
            x_t: Noisy images at timestep t
        """

        # Get parameters at timestep t and broadcast
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        # Compute x_t from x_0 and noise
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t

    def backward(self, x_t, t, noise_prediction):
        """
        Args:
            x_t: Noisy images at timestep t [batch_size, channels, height, width]
            t: Current timestep (scalar)
            noise_prediction: Predicted noise from model [batch_size, channels, height, width]

        Returns:
            x_{t-1}: Denoised images at previous timestep
        """

        # Get parameters at timestep t
        beta_t = self.betas[t]
        sqrt_beta_t = self.sqrt_betas[t]
        sqrt_alpha_t = self.sqrt_alphas[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]

        if t > 0:
            sqrt_one_minus_alpha_bar_t_minus_one = self.sqrt_one_minus_alpha_bars[t - 1]
            sigma = (sqrt_one_minus_alpha_bar_t_minus_one / sqrt_one_minus_alpha_bar_t) * sqrt_beta_t
            epsilon = torch.randn_like(x_t)
        else:
            sigma = 0
            epsilon = 0

        # Compute x_{t-1}
        prev = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * noise_prediction) / sqrt_alpha_t + sigma * epsilon

        return prev
    
    def backward_ddim(self, cfg, x_t, t, noise_prediction):
        """
        DDIM backward step.
        Uses a non-Markovian diffusion process that allows faster sampling by skipping timesteps.

        Args:
            cfg: Configuration object containing eta parameter (controls stochasticity)
            x_t: Noisy images at timestep t [batch_size, channels, height, width]
            t: Current timestep (scalar)
            noise_prediction: Predicted noise from model [batch_size, channels, height, width]

        Returns:
            x_{t-1}: Denoised images at previous timestep 
        """
        
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]

        x_0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_prediction) / sqrt_alpha_bar_t

        if t > 0:
            alpha_bar_t_minus_one = self.alpha_bars[t-1]
            sqrt_alpha_bar_t_minus_one = self.sqrt_alpha_bars[t-1]
            sqrt_one_minus_alpha_bar_t_minus_one = self.sqrt_one_minus_alpha_bars[t-1]
            sigma_t = cfg.eta * (sqrt_one_minus_alpha_bar_t_minus_one / sqrt_one_minus_alpha_bar_t) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_minus_one)
        else:
            return x_0

        epsilon = torch.randn_like(x_t)

        prev = sqrt_alpha_bar_t_minus_one * x_0 + torch.sqrt(1 - alpha_bar_t_minus_one - sigma_t ** 2) * noise_prediction + sigma_t * epsilon

        return prev

    def to(self, device):
        """Move all parameters to device"""
        self.betas = self.betas.to(device)
        self.sqrt_betas = self.sqrt_betas.to(device)
        self.alphas = self.alphas.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def get_name(self):
        return "DDPM"

    def get_noise(self, imgs, device, clamp_a=None):
        return torch.randn_like(imgs).to(device)

    def get_loss(self):
        return torch.nn.functional.mse_loss

    def get_per_sample_loss(self):
        """Returns per-sample loss (no batch reduction) for bucket tracking."""
        return lambda pred, target: ((pred - target) ** 2).mean(dim=(1, 2, 3))


class DLPM(DiffusionProcess):
    """Denoising Levy Probabilistic Model (DLPM) implementation"""

    def __init__(self, num_time_steps=1000, alpha=1.7):
        """
        Args:
            num_time_steps: Number of diffusion steps
            alpha: Tail index for alpha-stable distribution (1 < alpha <= 2)
        """

        self.num_time_steps = num_time_steps
        self.alpha = alpha
        self.betas = self._cosine_beta_schedule(num_time_steps)
        self.gammas = torch.pow(1 - self.betas, 1.0 / alpha)
        self.gamma_bars = torch.cumprod(self.gammas, dim=0)
        self.sigmas = torch.pow(1 - torch.pow(self.gammas, alpha), 1.0 / alpha)
        self.sigma_bars = torch.pow(1 - torch.pow(self.gamma_bars, alpha), 1.0 / alpha)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Args:
            timesteps: Number of diffusion steps
            s: Small offset to prevent beta from being too small near t=0

        Returns:
            betas: Beta values for each timestep
        """

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return torch.clip(betas, 0.001, 0.999)

    def sample_alpha_stable(self, size, device='cpu', clamp_a=None):
        """
        Sample from alpha/2-stable distribution using scipy.

        Args:
            size: Tuple specifying the shape of samples (e.g., (batch_size, timesteps)
            device: torch device
            clamp_a: Maximum value to clamp samples (None = no clamping)

        Returns:
            samples: Tensor of alpha/2-stable samples with the specified shape
        """
        alpha_half = self.alpha / 2.0
        c_A = np.cos(np.pi * self.alpha / 4.0) ** (2.0 / self.alpha)

        # scipy.stats.levy_stable parameters:
        # alpha: stability parameter (α/2)
        # beta: skewness parameter (1 for totally skewed to right)
        # loc: location (0)
        # scale: scale parameter (c_A)
        samples = levy_stable.rvs(alpha_half, beta=1, loc=0, scale=c_A, size=size)

        samples = torch.tensor(samples, dtype=torch.float32, device=device)

        if clamp_a is not None:
            samples = torch.clamp(samples, max=clamp_a)

        return samples

    def forward(self, x_0, noise, t):
        """
        Args:
            x_0: Clean images [batch_size, channels, height, width]
            noise: alpha-stable noise [batch_size, channels, height, width]
            t: Timestep indices [batch_size]

        Returns:
            x_t: Noisy images at timestep t
        """

        gamma_bar_t = self.gamma_bars[t]
        sigma_bar_t = self.sigma_bars[t]
        gamma_bar_t = gamma_bar_t[:, None, None, None]
        sigma_bar_t = sigma_bar_t[:, None, None, None]

        x_t = gamma_bar_t * x_0 + sigma_bar_t * noise

        return x_t

    def backward(self, x_t, t, noise_prediction, sigma_sq_all):
        """
        Args:
            x_t: Noisy images at timestep t [batch_size, channels, height, width]
            t: Current timestep (scalar)
            noise_prediction: Predicted noise from model [batch_size, channels, height, width]
            sigma_sq_all: Precomputed sigma_sq_all values [T, batch_size] for efficiency

        Returns:
            x_{t-1}: Denoised images at previous timestep
        """
        gamma_t = self.gammas[t]
        sigma_bar_t = self.sigma_bars[t]

        sigma_sq_1_to_t = sigma_sq_all[t]

        if t > 0:
            sigma_sq_1_to_t_minus_1 = sigma_sq_all[t - 1]

            gamma_t_sq = gamma_t ** 2
            gamma_coeff = 1 - (gamma_t_sq * sigma_sq_1_to_t_minus_1) / sigma_sq_1_to_t
            variance_t_minus_1 = gamma_coeff * sigma_sq_1_to_t_minus_1

            # Reshape for broadcasting with [batch_size, C, H, W]
            gamma_coeff = gamma_coeff[:, None, None, None]
            variance_t_minus_1 = variance_t_minus_1[:, None, None, None]

            gaussian_noise = torch.randn_like(x_t)
            noise_term = torch.sqrt(variance_t_minus_1) * gaussian_noise

        else:

            gamma_coeff = 1
            noise_term = 0

        prev = (x_t - gamma_coeff * sigma_bar_t * noise_prediction) / gamma_t + noise_term

        return prev

    def _precompute_all_sigma_sq(self, a_samples, device):
        """
        Args:
            a_samples: alpha/2-stable samples [batch_size, T] for all timesteps
            device: torch device

        Returns:
            sigma_sq_all: [T, batch_size] precomputed sigma_sq values for all timesteps
        """
        batch_size = a_samples.shape[0]
        T = a_samples.shape[1]

        sigma_sq_all = torch.zeros(T, batch_size, device=device)

        # Use recurrence formula:
        for t in range(T):
            if t == 0:
                sigma_sq_all[0] = self.sigmas[0]**2 * a_samples[:, 0]
            else:
                sigma_sq_all[t] = self.sigmas[t]**2 * a_samples[:, t] + self.gammas[t]**2 * sigma_sq_all[t-1]

        return sigma_sq_all

    def to(self, device):
        """Move all parameters to device"""
        self.betas = self.betas.to(device)
        self.gammas = self.gammas.to(device)
        self.gamma_bars = self.gamma_bars.to(device)
        self.sigmas = self.sigmas.to(device)
        self.sigma_bars = self.sigma_bars.to(device)
        return self

    def get_name(self):
        """Return the name of the diffusion process"""
        return "DLPM"

    def get_noise(self, imgs, device, clamp_a=None):
        """Generate alpha-stable noise samples for training target"""
        return torch.sqrt(self.sample_alpha_stable(size=(imgs.shape[0], 1, 1, 1), device=device, clamp_a=clamp_a)) * torch.randn_like(imgs).to(device)

    def get_loss(self):
        return lambda pred, target: torch.sqrt(((pred - target) ** 2).sum(dim=(1, 2, 3))).mean()

    def get_per_sample_loss(self):
        """Returns per-sample loss (no batch reduction) for bucket tracking."""
        return lambda pred, target: torch.sqrt(((pred - target) ** 2).sum(dim=(1, 2, 3)))

