#!/bin/bash
# LSF batch script to run FID batch size testing

# Job parameters
#BSUB -m comp02             # Force run on comp02 node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 8                 # Number of CPU cores (for data loading)
#BSUB -gpu num=1           # Request 1 GPU
#BSUB -M 8192              # Memory in MB (8 GB)
#BSUB -W 24:00             # Max time HH:MM (24 hours)
#BSUB -J fid_batch_size_test   # Job name
#BSUB -o output_fid_batch_size_test_%J.txt    # Standard output file (%J = job ID)
#BSUB -e error_fid_batch_size_test_%J.txt     # Error output file (%J = job ID)

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

# Run the FID batch size testing script
python -m tests.ddpm.fid_batch_size_test

# Print job completion info
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
