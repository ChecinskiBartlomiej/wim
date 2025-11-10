"""
FID batch size testing for DDPM CIFAR-10.

For each fid_batch_size, trains model for 2 epochs from scratch.
FID is calculated at epoch 2 automatically by the training function.
Monitors GPU and CPU utilization and measures total time.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ddpm_dlpm.ddpm_cifar10.config import CONFIG
from ddpm_dlpm.training import train
from tests.monitor import SystemMonitor


def test_fid_batch_sizes():
    """Test different FID batch sizes."""

    #fid_batch_sizes = [32, 64, 128, 256, 512]
    fid_batch_sizes = [64, 128, 256, 512]
    results = []

    print("="*70)
    print("FID BATCH SIZE TESTING FOR DDPM CIFAR-10")
    print("="*70)
    print(f"Testing FID batch sizes: {fid_batch_sizes}")
    print("="*70)

    # Set training parameters
    CONFIG.num_epochs = 2
    CONFIG.checkpoint_epochs = [2]
    CONFIG.num_fid_images = 640

    for fid_batch_size in fid_batch_sizes:
        print(f"\n{'='*70}")
        print(f"TESTING FID BATCH SIZE: {fid_batch_size}")
        print(f"{'='*70}\n")

        # Set fid_batch_size and directories for this test
        CONFIG.fid_batch_size = fid_batch_size
        CONFIG.model_dir = Path(f"outputs/fid_batch_size_test/fid_bs{fid_batch_size}/model")
        CONFIG.outputs_dir = Path(f"outputs/fid_batch_size_test/fid_bs{fid_batch_size}/outputs")

        # Start system monitoring
        monitor = SystemMonitor(interval=2.0)
        monitor.start()

        start_time = time.time()

        try:
            # Train model for 2 epochs (FID calculated at epoch 2)
            train(CONFIG, optimizer_name="AdamW")

            elapsed_time = time.time() - start_time
            max_mem, max_cpu_avg, max_cpu_core = monitor.stop()

            results.append({
                'fid_batch_size': fid_batch_size,
                'elapsed_time': elapsed_time,
                'max_memory_mb': max_mem,
                'max_cpu_avg': max_cpu_avg,
                'max_cpu_core': max_cpu_core,
                'status': 'success'
            })

            print(f"\n{'='*70}")
            print(f"FID BATCH SIZE {fid_batch_size} COMPLETED")
            print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"Max GPU memory: {max_mem} MB | CPU Avg: {max_cpu_avg:.1f}% | CPU Max core: {max_cpu_core:.1f}%")
            print(f"{'='*70}\n")

        except RuntimeError as e:
            elapsed_time = time.time() - start_time
            monitor.stop()

            if "out of memory" in str(e).lower():
                print(f"\n{'='*70}")
                print(f"FID BATCH SIZE {fid_batch_size} FAILED - OUT OF MEMORY")
                print(f"{'='*70}\n")
                results.append({
                    'fid_batch_size': fid_batch_size,
                    'elapsed_time': elapsed_time,
                    'status': 'OOM'
                })
            else:
                print(f"\n{'='*70}")
                print(f"FID BATCH SIZE {fid_batch_size} FAILED")
                print(f"Error: {e}")
                print(f"{'='*70}\n")
                results.append({
                    'fid_batch_size': fid_batch_size,
                    'elapsed_time': elapsed_time,
                    'status': 'failed',
                    'error': str(e)
                })

        except Exception as e:
            elapsed_time = time.time() - start_time
            monitor.stop()

            print(f"\n{'='*70}")
            print(f"FID BATCH SIZE {fid_batch_size} FAILED")
            print(f"Error: {e}")
            print(f"{'='*70}\n")
            results.append({
                'fid_batch_size': fid_batch_size,
                'elapsed_time': elapsed_time,
                'status': 'failed',
                'error': str(e)
            })

    # Print summary
    print("\n" + "="*95)
    print("FID BATCH SIZE TEST SUMMARY")
    print("="*95)
    print(f"{'BS':<6} {'Status':<10} {'Time(s)':<10} {'Time(m)':<10} {'GPU Mem':<10} {'CPU Avg%':<10} {'CPU Max%':<10}")
    print("-"*95)

    for result in results:
        bs = result['fid_batch_size']
        status = result['status']
        elapsed = result['elapsed_time']

        if status == 'success':
            mem = f"{result['max_memory_mb']}"
            cpu_avg = f"{result['max_cpu_avg']:.1f}"
            cpu_max = f"{result['max_cpu_core']:.1f}"
        else:
            mem = "N/A"
            cpu_avg = "N/A"
            cpu_max = "N/A"

        print(f"{bs:<6} {status:<10} {elapsed:<10.2f} {elapsed/60:<10.2f} {mem:<10} {cpu_avg:<10} {cpu_max:<10}")

    print("="*95)

    # Save results
    results_dir = Path("outputs/fid_batch_size_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "fid_batch_size_results.txt"

    with open(results_file, 'w') as f:
        f.write("FID BATCH SIZE TEST RESULTS\n")
        f.write("="*95 + "\n\n")
        f.write(f"Number of FID images: {CONFIG.num_fid_images}\n")
        f.write(f"Training epochs: {CONFIG.num_epochs}\n")
        f.write(f"FID calculated at epoch: {CONFIG.checkpoint_epochs[0]}\n\n")
        f.write(f"{'BS':<6} {'Status':<10} {'Time(s)':<10} {'Time(m)':<10} {'GPU Mem':<10} {'CPU Avg%':<10} {'CPU Max%':<10}\n")
        f.write("-"*95 + "\n")

        for result in results:
            bs = result['fid_batch_size']
            status = result['status']
            elapsed = result['elapsed_time']

            if status == 'success':
                mem = f"{result['max_memory_mb']}"
                cpu_avg = f"{result['max_cpu_avg']:.1f}"
                cpu_max = f"{result['max_cpu_core']:.1f}"
            else:
                mem = "N/A"
                cpu_avg = "N/A"
                cpu_max = "N/A"

            f.write(f"{bs:<6} {status:<10} {elapsed:<10.2f} {elapsed/60:<10.2f} {mem:<10} {cpu_avg:<10} {cpu_max:<10}\n")

            if status == 'failed' and 'error' in result:
                f.write(f"  Error: {result['error']}\n")

        f.write("="*95 + "\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    test_fid_batch_sizes()
