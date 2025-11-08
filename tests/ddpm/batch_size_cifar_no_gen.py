"""
Batch size testing for DDPM CIFAR-10 training without image generation.

Tests different batch sizes (16, 32, 64, 128, 192) by training for 4 epochs each,
measuring pure training time without generation overhead.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ddpm_dlpm.ddpm_cifar10.config import CONFIG
from ddpm_dlpm.training import train


def test_batch_sizes():
    """Test DDPM training with different batch sizes (no generation)."""

    batch_sizes = [16, 32, 64, 128, 192]
    results = []

    # Set test parameters
    CONFIG.num_epochs = 4
    CONFIG.checkpoint_epochs = []  # No generation

    print("="*70)
    print("BATCH SIZE TESTING FOR DDPM CIFAR-10 (NO GENERATION)")
    print("="*70)
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Epochs per test: {CONFIG.num_epochs}")
    print(f"Generation: DISABLED")
    print("="*70)
    print()

    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"TESTING BATCH SIZE: {batch_size}")
        print(f"{'='*70}\n")

        # Update config for this batch size
        CONFIG.batch_size = batch_size

        # Create batch-size-specific directories
        CONFIG.model_dir = Path(f"outputs/batch_size_test_no_gen/cifar10_bs{batch_size}")
        CONFIG.outputs_dir = Path(f"outputs/batch_size_test_no_gen/cifar10_bs{batch_size}")

        # Measure training time
        start_time = time.time()

        try:
            # Run training with AdamW optimizer
            train(CONFIG, optimizer_name="AdamW")

            elapsed_time = time.time() - start_time

            results.append({
                'batch_size': batch_size,
                'elapsed_time': elapsed_time,
                'status': 'success'
            })

            print(f"\n{'='*70}")
            print(f"BATCH SIZE {batch_size} COMPLETED")
            print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"{'='*70}\n")

        except Exception as e:
            elapsed_time = time.time() - start_time
            results.append({
                'batch_size': batch_size,
                'elapsed_time': elapsed_time,
                'status': 'failed',
                'error': str(e)
            })
            print(f"\n{'='*70}")
            print(f"BATCH SIZE {batch_size} FAILED")
            print(f"Error: {e}")
            print(f"{'='*70}\n")

    # Print summary
    print("\n" + "="*70)
    print("BATCH SIZE TEST SUMMARY (NO GENERATION)")
    print("="*70)
    print(f"{'Batch Size':<15} {'Status':<15} {'Time (s)':<15} {'Time (min)':<15}")
    print("-"*70)

    for result in results:
        status = result['status']
        elapsed = result['elapsed_time']
        print(f"{result['batch_size']:<15} {status:<15} {elapsed:<15.2f} {elapsed/60:<15.2f}")

    print("="*70)

    # Save results to file
    results_dir = Path("outputs/batch_size_test_no_gen")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "batch_size_results.txt"

    with open(results_file, 'w') as f:
        f.write("BATCH SIZE TEST RESULTS (NO GENERATION)\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Batch Size':<15} {'Status':<15} {'Time (s)':<15} {'Time (min)':<15}\n")
        f.write("-"*70 + "\n")

        for result in results:
            status = result['status']
            elapsed = result['elapsed_time']
            f.write(f"{result['batch_size']:<15} {status:<15} {elapsed:<15.2f} {elapsed/60:<15.2f}\n")

            if status == 'failed':
                f.write(f"  Error: {result['error']}\n")

        f.write("="*70 + "\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    test_batch_sizes()
