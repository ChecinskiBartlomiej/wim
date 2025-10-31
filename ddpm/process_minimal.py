import torch


class Process:
    """base class for forward and backward processes"""

    def __init__(self, num_time_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_time_steps = num_time_steps
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


class ForwardProcess(Process):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_noise(self, x_0, noise, t):

        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t = alpha_bar_t[:, None, None, None]

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t


class BackwardProcess(Process):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def denoise(self, x_t, t, noise_prediction):

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        if t > 0:
            alpha_bar_t_minus_one = self.alpha_bars[t - 1]
            sigma = torch.sqrt(((1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)) * beta_t)
            epsilon = torch.randn_like(x_t)
        else:
            sigma = 0
            epsilon = 0

        prev = (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_prediction) / torch.sqrt(alpha_t) + sigma * epsilon

        return prev
