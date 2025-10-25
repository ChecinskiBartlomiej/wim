#!/bin/bash
# LSF batch script to run MLP CPU training on WIM cluster

# Job parameters - CPU ONLY (no GPU)
#BSUB -m ALL               # Any available node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 4                 # Number of CPU cores (more cores for CPU training)
#BSUB -M 4096              # Memory in MB (4 GB)
#BSUB -W 1:30              # Max time HH:MM (90 minutes)
#BSUB -J mlp_cpu           # Job name (changed to indicate CPU)
#BSUB -o output_cpu_%J.txt # Standard output file (%J = job ID)
#BSUB -e error_cpu_%J.txt  # Error output file (%J = job ID)

# Print job start info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "MODE: CPU ONLY (no GPU requested)"

# Navigate to working directory
cd $HOME/test

# Activate virtual environment
source $HOME/mlp_env/bin/activate

# Run the Python script
python mlp_example_cpu.py

# Print job completion info
echo "Job finished at: $(date)"