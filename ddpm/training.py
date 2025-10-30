from ddpm.unet_utils import Unet
from ddpm.process import ForwardProcess
from ddpm.generate import generate
from ddpm.metrics import collect_gradient_stats, collect_weight_stats, save_weights_snapshot
from ddpm.visualization import plot_combined_stats, plot_generated_images

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np


def train(cfg, optimizer_name="Adam"):

    # Create optimizer-specific directories
    optimizer_model_dir = cfg.model_dir / optimizer_name
    optimizer_outputs_dir = cfg.outputs_dir / optimizer_name
    optimizer_model_dir.mkdir(parents=True, exist_ok=True)
    optimizer_outputs_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset_class(str(cfg.train_data_path))
    dataloader = DataLoader(dataset, cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Optimizer: {optimizer_name}\n")

    model = Unet(im_channels=cfg.im_channels).to(device)
    model_path = optimizer_model_dir / "ddpm_unet.pth"

    # Create optimizer based on name
    optimizer_config = cfg.optimizer_configs[optimizer_name]
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    criterion = torch.nn.MSELoss()

    fp = ForwardProcess()

    # Move all process parameters to device
    fp.betas = fp.betas.to(device)
    fp.sqrt_betas = fp.sqrt_betas.to(device)
    fp.alphas = fp.alphas.to(device)
    fp.sqrt_alphas = fp.sqrt_alphas.to(device)
    fp.alpha_bars = fp.alpha_bars.to(device)
    fp.sqrt_alpha_bars = fp.sqrt_alpha_bars.to(device)
    fp.sqrt_one_minus_alpha_bars = fp.sqrt_one_minus_alpha_bars.to(device)

    best_eval_loss = float("inf")
    epoch_losses = []

    # Gradient statistics tracking
    all_grad_norms = []
    all_zero_grad_pcts = []

    # Weight statistics tracking
    all_weight_norms = []
    all_zero_weight_pcts = []
    all_update_ratios = []

    for epoch in range(cfg.num_epochs):

        losses = []
        epoch_grad_stats = {
            "grad_norm": [],
            "zero_grad_percentage": []
        }
        epoch_weight_stats = {
            "weight_norm": [],
            "zero_weight_percentage": [],
            "update_ratio": []
        }

        model.train()

        for imgs in tqdm(dataloader):

            imgs = imgs.to(device)

            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)

            noisy_imgs = fp.add_noise(imgs, noise, t)

            optimizer.zero_grad()

            noise_pred = model(noisy_imgs, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            loss.backward()

            # Collect gradient statistics
            grad_stats = collect_gradient_stats(model)
            epoch_grad_stats["grad_norm"].append(grad_stats["grad_norm"])
            epoch_grad_stats["zero_grad_percentage"].append(grad_stats["zero_grad_pct"])

            # Save weights before update (for computing update ratio)
            old_weights = save_weights_snapshot(model)

            optimizer.step()

            # Collect weight statistics after update
            weight_stats = collect_weight_stats(model, old_weights)
            epoch_weight_stats["weight_norm"].append(weight_stats["weight_norm"])
            epoch_weight_stats["zero_weight_percentage"].append(weight_stats["zero_weight_pct"])
            epoch_weight_stats["update_ratio"].append(weight_stats["update_ratio"])

        mean_epoch_loss = np.mean(losses)
        epoch_losses.append(mean_epoch_loss)

        # Store epoch-averaged gradient statistics
        all_grad_norms.append(np.mean(epoch_grad_stats["grad_norm"]))
        all_zero_grad_pcts.append(np.mean(epoch_grad_stats["zero_grad_percentage"]))

        # Store epoch-averaged weight statistics
        all_weight_norms.append(np.mean(epoch_weight_stats["weight_norm"]))
        all_zero_weight_pcts.append(np.mean(epoch_weight_stats["zero_weight_percentage"]))
        all_update_ratios.append(np.mean(epoch_weight_stats["update_ratio"]))

        print(f"Epoch: {epoch+1} | Loss : {mean_epoch_loss:.4f}")

        if mean_epoch_loss < best_eval_loss:
            best_eval_loss = mean_epoch_loss
            torch.save(model.state_dict(), str(model_path))

        # Check if this is a checkpoint epoch
        if (epoch + 1) in cfg.checkpoint_epochs:
            print(f"\n{'='*60}")
            print(f"Checkpoint at epoch {epoch+1} - Generating images...")
            print(f"{'='*60}\n")

            checkpoint_model_path = optimizer_model_dir / f"ddpm_unet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), str(checkpoint_model_path))

            # Generate 64 images using current checkpoint model
            generated_imgs = []
            model.eval()
            with torch.no_grad():
                for i in tqdm(range(cfg.num_img_to_generate), desc=f"Generating images"):
                    x_t = generate(cfg, model=model)
                    x_t = x_t[0].numpy()  # Shape: (channels, H, W)
                    x_t = np.transpose(x_t, (1, 2, 0))  # Shape: (H, W, channels)
                    if cfg.im_channels == 1:
                        x_t = x_t.squeeze()  # Remove channel dimension for grayscale: (H, W)
                    x_t = 255 * x_t
                    generated_imgs.append(x_t.astype(np.uint8).flatten())

            # Plot generated images
            grid_plot_path = optimizer_outputs_dir / f"generated_images_epoch_{epoch+1}.png"
            plot_generated_images(generated_imgs, grid_plot_path, grid_size=(8, 8), cmap=cfg.cmap, im_channels=cfg.im_channels, img_size=cfg.img_size)

            print(f"Saved {cfg.num_img_to_generate} generated images to {grid_plot_path}")
            print(f"Checkpoint model saved to {checkpoint_model_path}")

            # Plot combined statistics up to current epoch (2x3 grid)
            epochs_so_far = list(range(1, epoch + 2))
            combined_plot_path = optimizer_outputs_dir / f"combined_stats_epoch_{epoch+1}.png"

            plot_combined_stats(
                epoch_losses,
                all_grad_norms,
                all_zero_grad_pcts,
                all_weight_norms,
                all_zero_weight_pcts,
                all_update_ratios,
                epochs_so_far,
                combined_plot_path,
                title_suffix=f" - {optimizer_name}"
            )

            print(f"Combined statistics plot saved to {combined_plot_path}\n")

            model.train()

    print(f"Done training")
