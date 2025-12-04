#!/bin/bash
# LSF batch script to run DLPM multi-GPU training with AdamW optimizer on WIM cluster

# Job parameters
#BSUB -m comp01              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 28                # Number of CPU cores (7 cores per GPU × 4 GPUs)
#BSUB -gpu num=4           # Request 4 GPUs
#BSUB -M 65536             # Memory in MB (64 GB, 16GB per GPU × 4)
#BSUB -W 99:00             # Max time HH:MM (99 hours)
#BSUB -J ddpm_adamw_multigpu   # Job name
#BSUB -o output_mnist_adamw_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error__mnist_adamw_%J.txt     # Error output file (%J = job ID)

# Print job start info
echo "=================================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "=================================================="

# Navigate to working directory
cd $HOME

# Activate virtual environment if you have one
source $HOME/ddpm_env/bin/activate

# Set CPU threads per GPU process to match job allocation (7 cores per GPU)
# By default torchrun sets OMP_NUM_THREADS=1, but we have 28 cores allocated
export OMP_NUM_THREADS=7

# Run the training script with torchrun for multi-GPU DDP
torchrun --nproc_per_node=4 --nnodes=1 -m ddpm_dlpm_multigpu.ddpm_mnist.train_AdamW

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
