from ddpm_dlpm_multigpu.unet import Unet
from ddpm_dlpm_multigpu.process import DDPM, DLPM
from ddpm_dlpm_multigpu.metrics import (
    collect_gradient_stats, collect_weight_stats, save_weights_snapshot,
    count_parameters, gather_training_stats
)
from ddpm_dlpm_multigpu.fid import calculate_fid_from_model_distributed
from ddpm_dlpm_multigpu.visualization import plot_combined_stats, plot_generated_images, plot_denoising_progress
from ddpm_dlpm_multigpu.ema import EMA

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import torch
import numpy as np
import os


def create_warmup_scheduler(optimizer, warmup_epochs=15, start_factor=0.2):
    """
    Create a learning rate warmup scheduler.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup (default: 15)
        start_factor: Starting LR multiplier (default: 0.2 = 1/5 of target LR)

    Returns:
        LambdaLR scheduler that linearly increases LR from start_factor to 1.0
    """
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from start_factor to 1.0 over warmup_epochs
            return start_factor + (1.0 - start_factor) * epoch / (warmup_epochs - 1)
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)


def train(cfg, optimizer_name="Adam"):

    # Get rank from environment and set device BEFORE initializing process group
    # This ensures NCCL knows the exact GPU mapping from the start
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()

    # Only rank 0 prints and creates directories
    if rank == 0:
        
        optimizer_model_dir = cfg.model_dir / optimizer_name
        optimizer_outputs_dir = cfg.outputs_dir / optimizer_name
        optimizer_model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_outputs_dir.mkdir(parents=True, exist_ok=True)

    else:

        optimizer_model_dir = cfg.model_dir / optimizer_name
        optimizer_outputs_dir = cfg.outputs_dir / optimizer_name

    dist.barrier()

    # Use DistributedSampler to split data across GPUs
    dataset = cfg.dataset_class(str(cfg.train_data_path), use_horizontal_flip=cfg.use_horizontal_flip)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    if rank == 0:
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Diffusion: {cfg.diffusion.__name__}")
        if hasattr(cfg, 'alpha'):
            print(f"Alpha: {cfg.alpha}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Horizontal flips: {cfg.use_horizontal_flip}")
        print(f"Batch size per GPU: {cfg.batch_size}")
        print(f"Effective batch size: {cfg.batch_size * world_size}")

    # Create diffusion process (DDPM/DLPM) and get its loss function
    diffusion_name = cfg.diffusion.__name__.lower()
    diffusion = cfg.diffusion().to(device)
    criterion = diffusion.get_loss()

    # Create Unet model and wrap it with DDP
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
        img_size=cfg.img_size,
        attention_resolutions=cfg.unet_attention_resolutions,
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    model_path = optimizer_model_dir / f"{diffusion_name}_unet.pth"

    # Count and display model parameters
    if rank == 0:
        param_counts = count_parameters(model.module)
        print(f"Total parameters: {param_counts['total_params']:,}")
        print(f"Trainable parameters: {param_counts['trainable_params']:,}\n")

    # Initialize EMA with decay=0.9999
    ema = EMA(model.module, decay=0.9999)
    if rank == 0:
        print(f"EMA initialized with decay=0.9999\n")

    # Create optimizer based on name
    optimizer_config = cfg.optimizer_configs[optimizer_name]
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if cfg.use_scheduler:
        scheduler = create_warmup_scheduler(optimizer, warmup_epochs=15, start_factor=0.2)

        if rank == 0:
            target_lr = optimizer_config["lr"]
            print(f"Learning rate warmup scheduler created:")
            print(f"  - Warmup epochs: 15")
            print(f"  - Start LR: {target_lr * 0.2:.2e} (0.2x target)")
            print(f"  - Target LR: {target_lr:.2e}")
            print()

    # Initialize loss value and define gradient thresholds
    best_eval_loss = float("inf")
    grad_thresholds = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]

    # Track global statistics
    if rank == 0:
        epoch_losses = []
        all_grad_norms = []
        all_grad_below_thresh = {thresh: [] for thresh in grad_thresholds}
        all_weight_norms = []
        all_zero_weight_pcts = []
        all_update_ratios = []
        all_fid_scores = []
        fid_file = optimizer_outputs_dir / "fid_scores.txt"

    # Checkpoint resumption
    start_epoch = 0
    if cfg.resume_checkpoint is not None:
        if rank == 0:
            print(f"{'='*70}")
            print(f"LOADING CHECKPOINT FOR RESUMPTION")
            print(f"{'='*70}")
            print(f"Checkpoint path: {cfg.resume_checkpoint}\n")

        checkpoint = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)

        # Restore model, EMA, optimizer states
        model.module.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler state if scheduler is enabled
        if cfg.use_scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Set starting epoch (resume from next epoch after checkpoint)
        start_epoch = checkpoint['epoch'] + 1

        # Restore statistics for rank 0 (for continuous plots)
        if rank == 0:
            epoch_losses = checkpoint.get('epoch_losses', [])
            all_grad_norms = checkpoint.get('all_grad_norms', [])
            all_grad_below_thresh = checkpoint.get('all_grad_below_thresh', {thresh: [] for thresh in grad_thresholds})
            all_weight_norms = checkpoint.get('all_weight_norms', [])
            all_zero_weight_pcts = checkpoint.get('all_zero_weight_pcts', [])
            all_update_ratios = checkpoint.get('all_update_ratios', [])
            all_fid_scores = checkpoint.get('all_fid_scores', [])

            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Starting training at epoch {start_epoch}")
            print(f"Previous loss: {checkpoint['loss']:.6f}")
            print(f"Loaded {len(epoch_losses)} epoch statistics")
            print(f"Loaded {len(all_fid_scores)} FID scores")
            print(f"{'='*70}\n")
        dist.barrier()

    # Training loop
    for epoch in range(start_epoch, start_epoch + cfg.num_epochs):

        # Set epoch for DistributedSampler (ensures proper shuffling)
        sampler.set_epoch(epoch)

        # Create empty data structures for storing statistics per epoch
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

        # Iterate through batches
        pbar = tqdm(dataloader) if rank == 0 else dataloader
        for imgs in pbar:

            # Do forward and make prediction of noise added, calculate loss and gradients
            imgs = imgs.to(device)
            noise = diffusion.get_noise(imgs, device, clamp_a=cfg.clamp_a)
            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)
            noisy_imgs = diffusion.forward(imgs, noise, t)
            optimizer.zero_grad()
            noise_pred = model(noisy_imgs, t)

            # Clamp noise predictions if clamp_eps is set (DLPM stability)
            if cfg.clamp_eps is not None:
                noise_pred = torch.clamp(noise_pred, -cfg.clamp_eps, cfg.clamp_eps)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()

            # Gradient clipping if grad_clip is set (DLPM stability)
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # Collect gradient statistics
            grad_stats = collect_gradient_stats(model.module, thresholds=grad_thresholds)
            epoch_grad_stats["grad_norm"].append(grad_stats["grad_norm"])
            for thresh in grad_thresholds:
                epoch_grad_stats["grad_below_thresh"][thresh].append(grad_stats["grad_below_thresh"][thresh])

            # Save weights before update (for computing update ratio) do optimizer step and collect weights statistics
            old_weights = save_weights_snapshot(model.module)
            optimizer.step()
            weight_stats = collect_weight_stats(model.module, old_weights)
            epoch_weight_stats["weight_norm"].append(weight_stats["weight_norm"])
            epoch_weight_stats["zero_weight_percentage"].append(weight_stats["zero_weight_pct"])
            epoch_weight_stats["update_ratio"].append(weight_stats["update_ratio"])

            # Update EMA
            ema.update(model.module)

        # Compute epoch averages for this rank
        local_loss = np.mean(losses)
        local_grad_norm = np.mean(epoch_grad_stats["grad_norm"])
        local_grad_below_thresh = {thresh: np.mean(epoch_grad_stats["grad_below_thresh"][thresh]) for thresh in grad_thresholds}
        local_weight_norm = np.mean(epoch_weight_stats["weight_norm"])
        local_zero_weight_pct = np.mean(epoch_weight_stats["zero_weight_percentage"])
        local_update_ratio = np.mean(epoch_weight_stats["update_ratio"])

        # Gather statistics from all ranks to rank 0 and compute global averages
        global_stats = gather_training_stats(
            local_loss=local_loss,
            local_grad_norm=local_grad_norm,
            local_grad_below_thresh=local_grad_below_thresh,
            local_weight_norm=local_weight_norm,
            local_zero_weight_pct=local_zero_weight_pct,
            local_update_ratio=local_update_ratio,
            grad_thresholds=grad_thresholds,
            rank=rank,
            world_size=world_size,
            device=device
        )
        
        if rank == 0:
            # Store global statistics
            epoch_losses.append(global_stats['loss'])
            all_grad_norms.append(global_stats['grad_norm'])
            for thresh in grad_thresholds:
                all_grad_below_thresh[thresh].append(global_stats['grad_below_thresh'][thresh])
            all_weight_norms.append(global_stats['weight_norm'])
            all_zero_weight_pcts.append(global_stats['zero_weight_pct'])
            all_update_ratios.append(global_stats['update_ratio'])

            print(f"Epoch: {epoch+1} | Loss: {global_stats['loss']:.4f} (averaged across {world_size} GPUs)")

            # Check if this is the best loss if yes save checkpoint
            if global_stats['loss'] < best_eval_loss:

                best_eval_loss = global_stats['loss']
                best_model_dict = {
                    'model_state_dict': model.module.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': global_stats['loss']
                }
                if cfg.use_scheduler:
                    best_model_dict['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(best_model_dict, str(model_path))

        # Synchronize all ranks after epoch statistics gathering
        dist.barrier()

        # Check if this is an image checkpoint epoch
        if (epoch + 1) in cfg.image_checkpoint_epochs:
            # Only rank 0 generates images and saves checkpoints
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Image checkpoint at epoch {epoch+1} - Generating images...")
                print(f"{'='*60}\n")

                checkpoint_model_path = optimizer_model_dir / f"{diffusion_name}_unet_epoch_{epoch+1}.pth"
                checkpoint_dict = {
                    'model_state_dict': model.module.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': global_stats['loss'],
                    # Save statistics for continuous plots when resuming
                    'epoch_losses': epoch_losses,
                    'all_grad_norms': all_grad_norms,
                    'all_grad_below_thresh': all_grad_below_thresh,
                    'all_weight_norms': all_weight_norms,
                    'all_zero_weight_pcts': all_zero_weight_pcts,
                    'all_update_ratios': all_update_ratios,
                    'all_fid_scores': all_fid_scores
                }

                if cfg.use_scheduler:
                    checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()

                torch.save(checkpoint_dict, str(checkpoint_model_path))

                # Generate all images at once with intermediate steps (batch generation), use EMA weights for generation
                model.eval()
                ema.apply_shadow(model.module)
                with torch.no_grad():

                    all_intermediate_images, all_timesteps = diffusion.generate(
                        cfg,
                        model=model.module,
                        return_intermediate=True,
                        intermediate_step=cfg.denoising_timestep_interval,
                        batch_size=cfg.num_img_to_generate
                    )

                    # Extract final images for grid plot
                    generated_imgs = []
                    for intermediate_imgs in all_intermediate_images:
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

                # Restore training weights after generating 25 images
                ema.restore(model.module)
                model.train()

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

            # Synchronize gpus
            dist.barrier()

        # Check if this is a FID checkpoint epoch
        if (epoch + 1) in cfg.fid_checkpoint_epochs:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"FID checkpoint at epoch {epoch+1} - Calculating FID...")
                print(f"{'='*60}\n")

            # Calculate FID score using ALL GPUs, use EMA weights for generation of images
            model.eval()
            ema.apply_shadow(model.module)

            with torch.no_grad():
                fid_score = calculate_fid_from_model_distributed(
                    diffusion,
                    model.module,
                    dataset,
                    cfg,
                    rank=rank,
                    world_size=world_size,
                    device=device
                )

            # Only rank 0 saves FID score (other ranks get None)
            if rank == 0 and fid_score is not None:
                all_fid_scores.append(fid_score)
                with open(fid_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}: FID = {fid_score:.2f}\n")
                print(f"FID results appended to: {fid_file}\n")

            dist.barrier()

            # All ranks restore training weights and switch back to train mode
            ema.restore(model.module)
            model.train()

        # Step the learning rate scheduler at the end of each epoch (if enabled)
        if cfg.use_scheduler:
            scheduler.step()

    if rank == 0:
        print(f"Done training")

    # Clean up distributed process group
    dist.destroy_process_group()
