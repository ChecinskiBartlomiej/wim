import torch
import torch.nn as nn

def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int):

    assert t_emb_dim % 2 == 0

    factor = 2 * torch.arange(start=0, end=t_emb_dim//2, dtype=torch.float32, device=time_steps.device) / t_emb_dim
    factor = 10000 ** factor
    t_emb = time_steps[:, None]
    t_emb = t_emb / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)

    return t_emb

class NormActConv(nn.Module):
    """Perform GroupNorm, Activation, Convolution"""

    def __init__(self,
                in_channels: int,
                out_channels: int,
                num_groups: int=8,
                kernel_size: int=3,
                norm: bool=True,
                act: bool=True):

        super().__init__()

        self.g_norm = nn.GroupNorm(num_groups, in_channels) if norm is True else nn.Identity()

        self.act = nn.SiLU() if act is True else nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x