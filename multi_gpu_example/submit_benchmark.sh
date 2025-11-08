#!/bin/bash
# LSF batch script to run GPU speed benchmark on WIM supercomputer

# Job parameters - request 4 GPUs (needed for the 4-GPU test)
#BSUB -m comp02              # Request A100 GPU nodes
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 16                # Number of CPU cores (4 per GPU for data loading)
#BSUB -gpu num=4           # Request 4 GPUs
#BSUB -M 8192             # Memory in MB (32 GB)
#BSUB -W 3:00              # Max time HH:MM (1 hour)
#BSUB -J gpu_benchmark     # Job name
#BSUB -o output_benchmark_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error_benchmark_%J.txt     # Error output file (%J = job ID)

echo "======================================================================"
echo "GPU SPEED BENCHMARK JOB"
echo "======================================================================"
echo "Job ID: $LSB_JOBID"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "======================================================================"
echo ""

# Navigate to home directory
cd $HOME

# Activate virtual environment (customize the environment name!)
# Make sure this venv has torch installed (torchvision not needed - uses synthetic data)
source $HOME/ddpm_env/bin/activate

# Run the Python benchmark script
# It will handle both tests (1 GPU and 4 GPUs) internally
python -m multi_gpu_example.run_gpu_benchmark

echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "======================================================================"
