import torch


class Process:
    """base class for forward and backward processes"""

    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_time_steps = num_time_steps
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.sqrt_betas = torch.sqrt(self.betas)
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)


class ForwardProcess(Process):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_noise(self, x_0, noise, t):

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]

        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t


class BackwardProcess(Process):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def denoise(self, x_t, t, noise_prediction):

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

        prev = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * noise_prediction) / sqrt_alpha_t + sigma * epsilon

        return prev
