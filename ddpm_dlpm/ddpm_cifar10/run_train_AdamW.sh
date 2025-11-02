#!/bin/bash
# LSF batch script to run DDPM training with AdamW optimizer on WIM cluster

# Job parameters
#BSUB -m A100              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 8                 # Number of CPU cores (for data loading)
#BSUB -gpu num=1           # Request 1 GPU
#BSUB -M 8192              # Memory in MB (16 GB)
#BSUB -W 24:00             # Max time HH:MM (24 hours)
#BSUB -J ddpm_cifar10_adamw # Job name
#BSUB -o output_cifar10_adamw_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error_cifar10_adamw_%J.txt     # Error output file (%J = job ID)

# Print job start info
echo "=================================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "=================================================="

# Navigate to working directory
cd $HOME/Desktop/wim

# Activate virtual environment if you have one
source $HOME/ddpm_env/bin/activate

# Run the training script
python -m ddpm.ddpm_cifar10.train_AdamW

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
