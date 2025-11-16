#!/bin/bash
# LSF batch script to run DLPM multi-GPU training with AdamW optimizer on WIM cluster

# Job parameters
#BSUB -m comp01              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 49                # Number of CPU cores (7 cores per GPU × 7 GPUs)
#BSUB -gpu num=7           # Request 7 GPUs
#BSUB -M 114688            # Memory in MB (112 GB, 16GB per GPU × 7)
#BSUB -W 99:00             # Max time HH:MM (99 hours)
#BSUB -J dlpm_mnist_adamw_multigpu # Job name
#BSUB -o output_mnist_adamw_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error_mnist_adamw_%J.txt     # Error output file (%J = job ID)

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

# Run the training script with torchrun for multi-GPU DDP
torchrun --nproc_per_node=7 --nnodes=1 -m ddpm_dlpm_multigpu.dlpm_mnist.train_AdamW

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
