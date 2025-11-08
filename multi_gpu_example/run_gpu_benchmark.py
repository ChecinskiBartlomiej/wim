"""
GPU Speed Benchmark - Compares training speed on 1 GPU vs 4 GPUs.

This script runs two training sessions:
1. Train for 2 epochs on 1 GPU
2. Train for 2 epochs on 4 GPUs
3. Compare the times

Usage:
    python multi_gpu_example/run_gpu_benchmark.py
"""

import subprocess
import time
import sys


def run_training_test(num_gpus, num_epochs=2):
    """
    Run training with specified number of GPUs and measure time.

    Args:
        num_gpus: Number of GPUs to use (1 or 4)
        num_epochs: Number of epochs to train

    Returns:
        elapsed_time: Time taken in seconds, or None if failed
    """
    print("=" * 70)
    print(f"TEST: TRAINING ON {num_gpus} GPU(s) FOR {num_epochs} EPOCHS")
    print("=" * 70)
    print()

    start_time = time.time()

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        "multi_gpu_example/train_multi_gpu.py",
        f"--num_epochs={num_epochs}"
    ]

    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed: {e}")
        return None

    elapsed_time = time.time() - start_time

    print()
    print(f"Training on {num_gpus} GPU(s) completed in {elapsed_time:.2f} seconds")
    print()

    return elapsed_time


def main():
    print("\n" + "=" * 70)
    print("GPU SPEED BENCHMARK")
    print("=" * 70)
    print("Comparing training speed: 1 GPU vs 4 GPUs")
    print("=" * 70)
    print()

    num_epochs = 2

    # Test 1: 1 GPU
    time_1gpu = run_training_test(num_gpus=1, num_epochs=num_epochs)

    if time_1gpu is None:
        print("Benchmark failed on 1 GPU test")
        sys.exit(1)

    # Wait a bit between tests
    print("Waiting 20 seconds before next test...")
    time.sleep(20)
    print()

    # Test 2: 4 GPUs
    time_4gpu = run_training_test(num_gpus=4, num_epochs=num_epochs)

    if time_4gpu is None:
        print("Benchmark failed on 4 GPU test")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Training time with 1 GPU:  {time_1gpu:.2f} seconds")
    print(f"Training time with 4 GPUs: {time_4gpu:.2f} seconds")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
