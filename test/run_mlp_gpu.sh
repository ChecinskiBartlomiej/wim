#!/bin/bash
# LSF batch script to run MLP training on WIM cluster

# Job parameters
#BSUB -m A100              # Request A100 GPU node
#BSUB -q normal            # Queue type (normal for batch jobs)
#BSUB -n 4                 # Number of CPU cores (2 is enough for this task)
#BSUB -gpu num=1           # Request 1 GPU
#BSUB -M 4096              # Memory in MB (4 GB)
#BSUB -W 1:30              # Max time HH:MM (90 minutes is plenty)
#BSUB -J mlp_gpu      # Job name
#BSUB -o output_gpu_%J.txt     # Standard output file (%J = job ID)
#BSUB -e error_gpu_%J.txt      # Error output file (%J = job ID)

# Print job start info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"a
echo "Job ID: $LSB_JOBID"

# Navigate to working directory
cd $HOME/test

# Activate virtual environment
source $HOME/mlp_env/bin/activate

# Run the Python script
python mlp_example_gpu.py

# Print job completion info
echo "Job finished at: $(date)"
