from ddpm_dlpm.unet import Unet
from ddpm_dlpm.process import DDPM, DLPM
from ddpm_dlpm.metrics import collect_gradient_stats, collect_weight_stats, save_weights_snapshot, count_parameters, calculate_fid_from_model
from ddpm_dlpm.visualization import plot_combined_stats, plot_generated_images, plot_denoising_progress

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

    dataset = cfg.dataset_class(str(cfg.train_data_path), use_horizontal_flip=cfg.use_horizontal_flip)
    dataloader = DataLoader(dataset, cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Diffusion: {cfg.diffusion.__name__}")
    if hasattr(cfg, 'alpha'):
        print(f"Alpha: {cfg.alpha}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Horizontal flips: {cfg.use_horizontal_flip}")

    diffusion = cfg.diffusion().to(device)

    model = Unet(
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
    ).to(device)

    # Count and display model parameters
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total_params']:,}")
    print(f"Trainable parameters: {param_counts['trainable_params']:,}\n")

    model_path = optimizer_model_dir / "ddpm_unet.pth"

    # Create optimizer based on name
    optimizer_config = cfg.optimizer_configs[optimizer_name]
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    criterion = diffusion.get_loss()

    best_eval_loss = float("inf")
    epoch_losses = []

    # Gradient statistics tracking with multiple thresholds
    grad_thresholds = [10e-11, 10e-10, 10e-9, 10e-8, 10e-7]
    all_grad_norms = []
    all_grad_below_thresh = {thresh: [] for thresh in grad_thresholds}  # Track each threshold separately

    # Weight statistics tracking
    all_weight_norms = []
    all_zero_weight_pcts = []
    all_update_ratios = []

    # FID tracking
    all_fid_scores = []
    fid_file = optimizer_outputs_dir / "fid_scores.txt"

    for epoch in range(cfg.num_epochs):

        losses = []
        epoch_grad_stats = {
            "grad_norm": [],
            "grad_below_thresh": {thresh: [] for thresh in grad_thresholds}
        }
        epoch_weight_stats = {
            "weight_norm": [],
            "zero_weight_percentage": [],
            "update_ratio": []
        }

        model.train()

        for imgs in tqdm(dataloader):

            imgs = imgs.to(device)

            noise = diffusion.get_noise(imgs, device)

            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)

            noisy_imgs = diffusion.forward(imgs, noise, t)

            optimizer.zero_grad()

            noise_pred = model(noisy_imgs, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            loss.backward()

            # Collect gradient statistics with multiple thresholds
            grad_stats = collect_gradient_stats(model, thresholds=grad_thresholds)
            epoch_grad_stats["grad_norm"].append(grad_stats["grad_norm"])
            for thresh in grad_thresholds:
                epoch_grad_stats["grad_below_thresh"][thresh].append(grad_stats["grad_below_thresh"][thresh])

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
        for thresh in grad_thresholds:
            all_grad_below_thresh[thresh].append(np.mean(epoch_grad_stats["grad_below_thresh"][thresh]))

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

            # Generate all images at once with intermediate steps (batch generation)
            model.eval()
            with torch.no_grad():
                # Generate all images in parallel
                all_intermediate_images, all_timesteps = diffusion.generate(
                    cfg,
                    model=model,
                    return_intermediate=True,
                    intermediate_step=cfg.denoising_timestep_interval,
                    batch_size=cfg.num_img_to_generate
                )

                # Extract final images for grid plot
                generated_imgs = []
                for intermediate_imgs in all_intermediate_images:
                    # Get last image (final denoised result)
                    final_img = intermediate_imgs[-1]
                    generated_imgs.append(final_img.flatten())

            # Plot generated images (final images only)
            grid_plot_path = optimizer_outputs_dir / f"generated_images_epoch_{epoch+1}.png"
            plot_generated_images(generated_imgs, grid_plot_path, cmap=cfg.cmap, im_channels=cfg.im_channels, img_size=cfg.img_size)

            print(f"Saved {cfg.num_img_to_generate} generated images to {grid_plot_path}")

            # Plot denoising progress (all intermediate timesteps)
            denoising_progress_path = optimizer_outputs_dir / f"denoising_progress_epoch_{epoch+1}.png"
            plot_denoising_progress(
                all_intermediate_images,
                all_timesteps,
                denoising_progress_path,
                cmap=cfg.cmap,
                im_channels=cfg.im_channels,
                img_size=cfg.img_size
            )

            print(f"Saved denoising progress visualization to {denoising_progress_path}")
            print(f"Checkpoint model saved to {checkpoint_model_path}")

            # Plot combined statistics up to current epoch (2x3 grid)
            epochs_so_far = list(range(1, epoch + 2))
            combined_plot_path = optimizer_outputs_dir / f"combined_stats_epoch_{epoch+1}.png"

            plot_combined_stats(
                epoch_losses,
                all_grad_norms,
                all_grad_below_thresh,
                all_weight_norms,
                all_zero_weight_pcts,
                all_update_ratios,
                epochs_so_far,
                combined_plot_path,
                title_suffix=f" - {optimizer_name}"
            )

            print(f"Combined statistics plot saved to {combined_plot_path}\n")

            # Calculate FID score
            fid_score = calculate_fid_from_model(diffusion, model, dataset, cfg, device=device)
            all_fid_scores.append(fid_score)

            # Save FID to single file (append mode)
            
            with open(fid_file, 'a') as f:
                f.write(f"Epoch {epoch+1}: FID = {fid_score:.2f}\n")
            print(f"FID results appended to: {fid_file}\n")

            model.train()

    print(f"Done training")
