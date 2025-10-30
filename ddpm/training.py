from ddpm.data import CustomMnistDataset
from ddpm.unet_utils import Unet
from ddpm.process import ForwardProcess
from ddpm.generate import generate

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


def train(cfg, optimizer_name="Adam"):

    # Create optimizer-specific directories
    optimizer_model_dir = cfg.model_dir / optimizer_name
    optimizer_outputs_dir = cfg.outputs_dir / optimizer_name
    optimizer_model_dir.mkdir(parents=True, exist_ok=True)
    optimizer_outputs_dir.mkdir(parents=True, exist_ok=True)

    train_csv = cfg.data_dir / "train.csv"
    mnist_ds = CustomMnistDataset(str(train_csv))
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Optimizer: {optimizer_name}\n")

    model = Unet().to(device)
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

        for imgs in tqdm(mnist_dl):

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
            all_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.flatten())

            if len(all_grads) > 0:
                all_grads = torch.cat(all_grads)

                grad_norm = torch.norm(all_grads).item()
                zero_grad_pct = (torch.sum(torch.abs(all_grads) < 1e-7).item() / all_grads.numel()) * 100

                epoch_grad_stats["grad_norm"].append(grad_norm)
                epoch_grad_stats["zero_grad_percentage"].append(zero_grad_pct)

            # Save weights before update (for computing update ratio)
            old_weights = {}
            for name, param in model.named_parameters():
                old_weights[name] = param.data.clone()

            optimizer.step()

            # Collect weight statistics after update
            all_weights = []
            update_ratios = []
            for name, param in model.named_parameters():
                weight = param.data.flatten()
                all_weights.append(weight)

                # Compute update ratio
                weight_update = param.data - old_weights[name]
                ratio = torch.abs(weight_update) / (torch.abs(old_weights[name]) + 1e-10)
                update_ratios.append(ratio.flatten())

            if len(all_weights) > 0:
                all_weights = torch.cat(all_weights)
                all_update_ratios = torch.cat(update_ratios)

                weight_norm = torch.norm(all_weights).item()
                zero_weight_pct = (torch.sum(torch.abs(all_weights) < 1e-7).item() / all_weights.numel()) * 100
                update_ratio = torch.mean(all_update_ratios).item()

                epoch_weight_stats["weight_norm"].append(weight_norm)
                epoch_weight_stats["zero_weight_percentage"].append(zero_weight_pct)
                epoch_weight_stats["update_ratio"].append(update_ratio)

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
                    x_t = 255 * x_t[0][0].numpy()
                    generated_imgs.append(x_t.astype(np.uint8))

            fig, axes = plt.subplots(8, 8, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(np.reshape(generated_imgs[i], (28, 28)), cmap="gray")
                ax.axis("off")

            plt.tight_layout()
            grid_plot_path = optimizer_outputs_dir / f"generated_images_epoch_{epoch+1}.png"
            plt.savefig(str(grid_plot_path), dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Saved {cfg.num_img_to_generate} generated images to {grid_plot_path}")
            print(f"Checkpoint model saved to {checkpoint_model_path}")

            # Plot combined statistics up to current epoch (2x3 grid)
            epochs_so_far = list(range(1, epoch + 2))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Loss, Grad Norm, Zero Grad %
            axes[0, 0].plot(epochs_so_far, epoch_losses, 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(epochs_so_far, all_grad_norms, 'g-o', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Gradient Norm', fontsize=12)
            axes[0, 1].set_title('Gradient Norm', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

            axes[0, 2].plot(epochs_so_far, all_zero_grad_pcts, 'r-o', linewidth=2, markersize=6)
            axes[0, 2].set_xlabel('Epoch', fontsize=12)
            axes[0, 2].set_ylabel('Zero Grad %', fontsize=12)
            axes[0, 2].set_title('Zero Gradient %', fontsize=14, fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3)

            # Row 2: Weight Norm, Zero Weight %, Update Ratio
            axes[1, 0].plot(epochs_so_far, all_weight_norms, 'c-o', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Weight Norm', fontsize=12)
            axes[1, 0].set_title('Weight Norm', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

            axes[1, 1].plot(epochs_so_far, all_zero_weight_pcts, 'm-o', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Zero Weight %', fontsize=12)
            axes[1, 1].set_title('Zero Weight %', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

            axes[1, 2].plot(epochs_so_far, all_update_ratios, 'orange', linewidth=2, markersize=6, marker='o')
            axes[1, 2].set_xlabel('Epoch', fontsize=12)
            axes[1, 2].set_ylabel('Update Ratio', fontsize=12)
            axes[1, 2].set_title('Weight Update Ratio', fontsize=14, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

            plt.tight_layout()
            combined_plot_path = optimizer_outputs_dir / f"combined_stats_epoch_{epoch+1}.png"
            plt.savefig(str(combined_plot_path), dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Combined statistics plot saved to {combined_plot_path}\n")

            model.train()

    print(f"Done training")
