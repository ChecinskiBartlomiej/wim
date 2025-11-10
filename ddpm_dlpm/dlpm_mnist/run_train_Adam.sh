#!/bin/bash
# LSF batch script to run DLPM training with Adam optimizer on WIM cluster

# Job parameters
#BSUB -m A100              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 7                 # Number of CPU cores (for data loading)
#BSUB -gpu num=1           # Request 1 GPU
#BSUB -M 16384             # Memory in MB (16 GB)
#BSUB -W 72:00             # Max time HH:MM (72 hours)
#BSUB -J dlpm_adam         # Job name
#BSUB -o output_adam_%J.txt     # Standard output file (%J = job ID)
#BSUB -e error_adam_%J.txt      # Error output file (%J = job ID)

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

# Run the training script
python -m ddpm_dlpm.dlpm_mnist.train_Adam

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
