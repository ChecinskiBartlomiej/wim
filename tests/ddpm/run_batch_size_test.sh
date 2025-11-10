#!/bin/bash
# LSF batch script to run batch size testing with image generation

# Job parameters
#BSUB -m comp01              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 8                 # Number of CPU cores (for data loading)
#BSUB -gpu num=1           # Request 1 GPU
#BSUB -M 8192             # Memory in MB (8 GB)
#BSUB -W 24:00             # Max time HH:MM (72 hours for all batch sizes)
#BSUB -J batch_size_test   # Job name
#BSUB -o output_batch_size_test_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error_batch_size_test_%J.txt     # Error output file (%J = job ID)

# Print job start info
echo "=================================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "=================================================="

# Navigate to working directory
cd $HOME

# Activate virtual environment
source $HOME/ddpm_env/bin/activate

# Run the batch size testing script
python -m tests.ddpm.batch_size_cifar

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
