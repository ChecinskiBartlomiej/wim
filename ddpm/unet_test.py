from ddpm.process import ForwardProcess as FP, BackwardProcess as BP
from ddpm.process_minimal import (
    ForwardProcess as FPMinimal,
    BackwardProcess as BPMinimal,
)
from ddpm.unet_utils import (
    get_time_embedding,
    NormActConv,
    TimeEmbedding,
    SelfAttentionBlock,
    Downsample,
    Upsample,
    DownC,
    MidC,
    UpC,
    Unet,
)
import torch

print("testing forward:")
forward = FP()

x_0 = torch.randn(4, 1, 28, 28).to("cuda")
noise = torch.randn(4, 1, 28, 28).to("cuda")
t_steps = torch.randint(0, 1000, (4,))
out = forward.add_noise(x_0, noise, t_steps)
print(out.shape)
print(out.device)

print("testing forward finished")

print("testing forward minimal")

forward_minimal = FPMinimal()
x_0 = torch.randn(4, 1, 28, 28).to("cuda")
noise = torch.randn(4, 1, 28, 28).to("cuda")
t_steps = torch.randint(0, 1000, (4,))
out = forward_minimal.add_noise(x_0, noise, t_steps)
print(out.shape)
print(out.device)

print("testing forward minimal finished")

print("testing backward:")

backward = BP()

x_t = torch.randn(1, 1, 28, 28).to("cuda")
t = torch.randint(0, 1000, (1,))
noise_pred = torch.randn(1, 1, 28, 28).to("cuda")
out = backward.denoise(x_t, t, noise_pred)
print(out.shape)
print(out.device)

print("testing backward finished")

print("testing backward minimal")

backward_minimal = BPMinimal()
x_t = torch.randn(1, 1, 28, 28).to("cuda")
t = torch.randint(0, 1000, (1,))
noise_pred = torch.randn(1, 1, 28, 28).to("cuda")
out = backward_minimal.denoise(x_t, t, noise_pred)
print(out.shape)
print(out.device)

print("testing backward minimal finished")

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

print("testing TimeEmbedding")

time_steps = torch.randint(0, 1000, (4,)).to("cuda")
t_emb_dim = 128

time_embedding = get_time_embedding(time_steps=time_steps, t_emb_dim=t_emb_dim)
time_proj = TimeEmbedding(n_out=256, t_emb_dim=t_emb_dim).to("cuda")

projected = time_proj(time_embedding)

print(projected.shape)
print(projected.device)

print("testing TimeEmbedding finished")

print("testing Self Attention Block")

selfattentionblock = SelfAttentionBlock(num_channels=32, num_groups=4, num_heads=4, norm=True).to("cuda")
x = torch.randn(10, 32, 28, 28).to("cuda")
out = selfattentionblock(x)
print(out.shape)
print(out.device)

print("testing Self Attention Block finished")

print("testing Downsampling")

downsampling_1 = Downsample(in_channels=64, out_channels=32, k=2, use_conv=True, use_mpool=True).to("cuda")
downsampling_2 = Downsample(in_channels=64, out_channels=32, k=2, use_conv=True, use_mpool=False).to("cuda")
downsampling_3 = Downsample(in_channels=64, out_channels=32, k=2, use_conv=False, use_mpool=True).to("cuda")
x = torch.randn(10, 64, 28, 28).to("cuda")

out_1 = downsampling_1(x)
out_2 = downsampling_2(x)
out_3 = downsampling_3(x)

print(out_1.shape)
print(out_1.device)

print(out_2.shape)
print(out_2.device)

print(out_3.shape)
print(out_3.device)

print("testing Downsampling finished")

print("testing Upsampling")

upsampling_1 = Upsample(in_channels=64, out_channels=32, k=2, use_conv=True, use_upsample=True).to("cuda")
upsampling_2 = Upsample(in_channels=64, out_channels=32, k=2, use_conv=True, use_upsample=False).to("cuda")
upsampling_3 = Upsample(in_channels=64, out_channels=32, k=2, use_conv=False, use_upsample=True).to("cuda")
x = torch.randn(10, 64, 28, 28).to("cuda")

out_1 = upsampling_1(x)
out_2 = upsampling_2(x)
out_3 = upsampling_3(x)

print(out_1.shape)
print(out_1.device)

print(out_2.shape)
print(out_2.device)

print(out_3.shape)
print(out_3.device)

print("testing Upsampling finished")

print("testing DownC")

down = DownC(in_channels=64, out_channels=128).to("cuda")
x = torch.randn(10, 64, 28, 28).to("cuda")
time_steps = torch.randint(0, 1000, (10,)).to("cuda")
t_emb_dim = 128
time_embedding = get_time_embedding(time_steps=time_steps, t_emb_dim=t_emb_dim)

out = down(x, time_embedding)
print(out.shape)
print(out.device)

print("testing DownC finished")

print("testing MidC")
mid = MidC(in_channels=64, out_channels=64).to("cuda")
out = mid(x, time_embedding)
print(out.shape)
print(out.device)
print("testing MidC finished")

print("testing UpC")

up = UpC(in_channels=64, out_channels=32).to("cuda")
down_out = torch.randn(10, 32, 56, 56).to("cuda")
out = up(x, down_out, time_embedding)
print(out.shape)
print(out.device)

print("testing UpC finished")

print("testing Unet")

x = torch.randn(10, 3, 28, 28).to("cuda")
t = torch.randint(0, 1000, (10,)).to("cuda")
unet = Unet(im_channels=3).to("cuda")
out = unet(x, t)
print(out.shape)
print(out.device)

print("testing Unet finished")
