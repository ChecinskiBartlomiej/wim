from ddpm.utils.process import ForwardProcess, BackwardProcess
import torch

print("testing forward:")
forward = ForwardProcess()

x_0 = torch.randn(4, 1, 28, 28)
t_steps = torch.randint(0, 1000, (4,))
out_forward = forward.add_noise(x_0, t_steps)
print(out_forward.shape)

print("testing backward:")
backward = BackwardProcess()

x_t = torch.randn(1, 1, 28, 28)
t = torch.randint(0, 1000, (1,))
noise_pred = torch.randn(1, 1, 28, 28)
out_backward = backward.denoise(x_t, t, noise_pred)
print(out_backward.shape)

print("ok")

