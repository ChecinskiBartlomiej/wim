"""
Multi-GPU training script using PyTorch DistributedDataParallel (DDP).

This script demonstrates how to train a model across multiple GPUs efficiently.

Key Concepts:
1. Each GPU runs a separate process (rank)
2. Each process has its own copy of the model
3. Gradients are automatically synchronized across all GPUs
4. Data is split across GPUs using DistributedSampler
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
import os
import sys
import time

from model import SimpleCNN, count_parameters


def setup_distributed(rank, world_size):
    """
    Initialize the distributed process group.

    Args:
        rank: Unique identifier for this process (0 to world_size-1)
              For 4 GPUs: rank will be 0, 1, 2, or 3
        world_size: Total number of processes (= number of GPUs)
                   For 4 GPUs: world_size = 4

    What happens here:
    - Sets up communication backend (NCCL is optimized for NVIDIA GPUs)
    - Each process finds the others and establishes communication channels
    - This allows gradients to be synchronized during training
    """
    # NCCL is the NVIDIA Collective Communications Library - fastest for GPU-to-GPU communication
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Uses environment variables set by torchrun
        world_size=world_size,
        rank=rank
    )


def cleanup_distributed():
    """
    Clean up the distributed process group after training.
    Closes all communication channels between processes.
    """
    dist.destroy_process_group()


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic random data for speed benchmarking.
    No need for real CIFAR-10 data - just generates random tensors!
    """
    def __init__(self, size=50000, image_size=(3, 32, 32), num_classes=10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random image and label
        image = torch.randn(*self.image_size)  # Random image
        label = torch.randint(0, self.num_classes, (1,)).item()  # Random label
        return image, label


def get_dataloaders(rank, world_size, batch_size):
    """
    Create data loaders for training and validation.

    Important: DistributedSampler ensures each GPU gets different data!

    Args:
        rank: This GPU's rank (0-3 for 4 GPUs)
        world_size: Total number of GPUs
        batch_size: Batch size PER GPU

    How data is split:
    - If you have 50,000 training images and 4 GPUs:
      - Each GPU gets 50,000/4 = 12,500 unique images
    - If batch_size=128:
      - TOTAL batch size = 128 * 4 = 512 images processed in parallel
      - Each GPU processes 128 images independently
      - Gradients are averaged across all 512 images
    """
    # Use synthetic random data for benchmarking (no real data needed!)
    # This removes data loading overhead and focuses on pure GPU speed
    train_dataset = SyntheticDataset(size=50000, image_size=(3, 32, 32), num_classes=10)
    test_dataset = SyntheticDataset(size=10000, image_size=(3, 32, 32), num_classes=10)

    # DistributedSampler: Splits data across GPUs
    # - shuffle=True: Shuffles data at start of each epoch
    # - set_epoch() must be called each epoch to get different shuffling
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,  # Total number of GPUs
        rank=rank,                 # This GPU's rank
        shuffle=True
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # DataLoader: batch_size is PER GPU, not total!
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,    # Use DistributedSampler instead of shuffle=True
        num_workers=4,             # Workers per GPU
        pin_memory=True            # Faster GPU transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, device, rank, epoch):
    """
    Train for one epoch.

    What happens during training with DDP:
    1. Each GPU processes its own batch independently (forward pass)
    2. Each GPU computes loss and gradients independently (backward pass)
    3. DDP automatically averages gradients across all GPUs
    4. Each GPU updates its model with the averaged gradients (optimizer.step)
    5. All models stay synchronized!
    """
    model.train()

    # Only show progress bar on rank 0 to avoid clutter
    if rank == 0:
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass - each GPU processes different data
        output = model(data)
        loss = criterion(output, target)

        # Backward pass - each GPU computes gradients
        loss.backward()

        # DDP magic happens here!
        # Before optimizer.step(), DDP automatically:
        # 1. Synchronizes gradients across all GPUs using all-reduce
        # 2. Averages them: final_grad = (grad_gpu0 + grad_gpu1 + grad_gpu2 + grad_gpu3) / 4
        # 3. Each GPU gets the same averaged gradient
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.reshape(pred.shape)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, test_loader, criterion, device, rank):
    """
    Validate the model.

    Note: Each GPU evaluates on its subset of validation data.
    We need to gather results from all GPUs to get true accuracy.
    """
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.reshape(pred.shape)).sum().item()
            total += target.size(0)

    # Convert to tensors for all-reduce
    test_loss = torch.tensor(test_loss).to(device)
    correct = torch.tensor(correct).to(device)
    total = torch.tensor(total).to(device)

    # Gather results from all GPUs
    # all_reduce with SUM: adds up values from all GPUs
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    # Compute global metrics
    avg_loss = test_loss.item() / (len(test_loader) * dist.get_world_size())
    accuracy = 100. * correct.item() / total.item()

    return avg_loss, accuracy


def main(rank, world_size, num_epochs=10):
    """
    Main training function that runs on each GPU.

    Args:
        rank: GPU identifier (0, 1, 2, 3 for 4 GPUs)
        world_size: Total number of GPUs (4)
        num_epochs: Number of epochs to train (default: 10)

    This function is called 4 times (once per GPU) by torchrun.
    Each call runs in a separate process with a different rank.
    """
    # Start timer (only on rank 0)
    if rank == 0:
        start_time = time.time()
        print("="*70)
        print("MULTI-GPU TRAINING WITH DISTRIBUTEDDATAPARALLEL")
        print("="*70)
        print(f"World size (number of GPUs): {world_size}")
        print(f"This is a learning example for multi-GPU training")
        print("="*70)

    # Setup distributed training
    setup_distributed(rank, world_size)

    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Configuration
    batch_size = 128  # Per GPU! Total batch size = 128 * 4 = 512
    learning_rate = 0.1

    if rank == 0:
        print(f"\nTraining configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Learning rate: {learning_rate}\n")

    # Create model
    model = SimpleCNN(num_classes=10).to(device)

    if rank == 0:
        print(f"Model: SimpleCNN")
        print(f"Total parameters: {count_parameters(model):,}\n")

    # Wrap model with DDP
    # This is the key step that enables multi-GPU training!
    # DDP will:
    # - Replicate the model on each GPU
    # - Automatically synchronize gradients during backward pass
    # - Keep all model copies in sync
    model = DDP(model, device_ids=[rank])

    # Create data loaders
    train_loader, test_loader, train_sampler = get_dataloaders(rank, world_size, batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # IMPORTANT: Set epoch for sampler to shuffle data differently each epoch
        train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, rank, epoch)

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device, rank)

        # Update learning rate
        scheduler.step()

        # Print results (only from rank 0)
        if rank == 0:
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}\n")

    # Save model (only from rank 0 to avoid conflicts)
    if rank == 0:
        save_dir = Path("multi_gpu_example/outputs")
        save_dir.mkdir(parents=True, exist_ok=True)
        # Access the underlying model with .module when using DDP
        torch.save(model.module.state_dict(), save_dir / "model_final.pth")
        print(f"\nModel saved to {save_dir / 'model_final.pth'}")

        # Print timing information
        elapsed_time = time.time() - start_time
        print("="*70)
        print("TRAINING COMPLETED!")
        print(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Time per epoch: {elapsed_time/num_epochs:.2f} seconds")
        print("="*70)

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    # These environment variables are set by torchrun
    # RANK: Global rank of this process (0-3 for 4 GPUs)
    # WORLD_SIZE: Total number of processes (4 for 4 GPUs)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Parse command line arguments
    num_epochs = 10  # Default
    for arg in sys.argv[1:]:
        if arg.startswith('--num_epochs='):
            num_epochs = int(arg.split('=')[1])

    main(rank, world_size, num_epochs)
