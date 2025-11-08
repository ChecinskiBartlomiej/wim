#!/bin/bash
# LSF batch script to run batch size testing without image generation

# Job parameters
#BSUB -m A100              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 8                 # Number of CPU cores (for data loading)
#BSUB -gpu num=1           # Request 1 GPU
#BSUB -M 8192            # Memory in MB (16 GB)
#BSUB -W 24:00             # Max time HH:MM (72 hours for all batch sizes)
#BSUB -J batch_size_test_no_gen   # Job name
#BSUB -o output_batch_size_test_no_gen_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error_batch_size_test_no_gen_%J.txt     # Error output file (%J = job ID)

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

# Run the batch size testing script (no generation)
python -m tests.ddpm.batch_size_cifar_no_gen

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
