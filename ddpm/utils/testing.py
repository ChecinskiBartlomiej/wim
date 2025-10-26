from ddpm.utils.process import ForwardProcess, BackwardProcess
from ddpm.utils.unet_utils import get_time_embedding, NormActConv
import torch

print("testing forward:")
forward = ForwardProcess()

x_0 = torch.randn(4, 1, 28, 28).to("cuda")
t_steps = torch.randint(0, 1000, (4,))
out_forward = forward.add_noise(x_0, t_steps)
print(out_forward.shape)
print(out_forward.device)

print("testing backward:")
backward = BackwardProcess()

x_t = torch.randn(1, 1, 28, 28).to("cuda")
t = torch.randint(0, 1000, (1,))
noise_pred = torch.randn(1, 1, 28, 28).to("cuda")
out_backward = backward.denoise(x_t, t, noise_pred)
print(out_backward.shape)
print(out_backward.device)

print("testing forward and backward finished")

print("testing time_embedding")

time_steps = torch.randint(0, 1000, (6,)).to("cuda")
t_emb_dim = 16

time_embedding = get_time_embedding(time_steps=time_steps, t_emb_dim=t_emb_dim)
print(time_embedding)
print(time_embedding.shape)
print(time_embedding.device)

print("testing time embedding finished")

print("testing NormActConv")

normactconv = NormActConv(in_channels=32, out_channels=8, num_groups=4, kernel_size=3, norm=True, act=True).to("cuda")
x = torch.randn(4, 32, 28, 28).to("cuda")
out = normactconv(x)
print(out)
print(out.shape)
print(out.device)

print("testing NormActConv finished")
