"""
System monitoring utilities for GPU and CPU utilization tracking.

Provides a background thread-based monitor for tracking GPU memory usage,
GPU utilization, and per-core CPU utilization during training and testing.
"""

import threading
import time
import subprocess
import psutil
import os


class SystemMonitor:
    """Monitor GPU and CPU utilization in a background thread."""

    def __init__(self, interval=2.0):
        """
        Initialize the system monitor.

        Args:
            interval: Monitoring interval in seconds (default: 2.0)
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.max_memory_used = 0
        self.max_cpu_avg = 0.0
        self.max_cpu_core = 0.0

    def _monitor(self):
        """Background monitoring function."""
        while self.running:
            try:
                # GPU stats using nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )

                memory_mb = 0
                gpu_util = 0
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        parts = lines[0].split(',')
                        memory_mb = int(parts[0].strip())
                        gpu_util = int(parts[1].strip())
                        self.max_memory_used = max(self.max_memory_used, memory_mb)

                # CPU stats (per-core breakdown)
                cpu_percents = psutil.cpu_percent(interval=1, percpu=True)

                # Calculate average across ALL cores (including zeros) for accurate system load
                cpu_avg = sum(cpu_percents) / len(cpu_percents)
                cpu_max_core = max(cpu_percents)

                self.max_cpu_avg = max(self.max_cpu_avg, cpu_avg)
                self.max_cpu_core = max(self.max_cpu_core, cpu_max_core)

                # Filter to only non-zero cores for DISPLAY only (cleaner output)
                active_cores = [(i, p) for i, p in enumerate(cpu_percents) if p > 0]
                cores_str = " ".join([f"C{i}:{p:.0f}%" for i, p in active_cores])
                active_count = len(active_cores)
                total_cores = len(cpu_percents)

                print(f"  [GPU] Mem: {memory_mb} MB | GPU: {gpu_util}% | [CPU] Avg: {cpu_avg:.1f}% | Max: {cpu_max_core:.0f}% | Active: {active_count}/{total_cores} | {cores_str}")

            except Exception as e:
                print(f"  [Monitor Error] {e}")

            time.sleep(self.interval)

    def start(self):
        """Start monitoring in a background thread."""
        self.running = True
        self.max_memory_used = 0
        self.max_cpu_avg = 0.0
        self.max_cpu_core = 0.0
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print("System monitoring started (GPU + CPU)...")

    def stop(self):
        """
        Stop monitoring and return maximum observed statistics.

        Returns:
            tuple: (max_memory_mb, max_cpu_avg_percent, max_cpu_core_percent)
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"Monitoring stopped. Max GPU mem: {self.max_memory_used} MB | CPU Avg: {self.max_cpu_avg:.1f}% | CPU Max core: {self.max_cpu_core:.1f}%")
        return self.max_memory_used, self.max_cpu_avg, self.max_cpu_core
