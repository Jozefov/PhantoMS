#!/bin/bash
#SBATCH --job-name=CutTreesValidation       # Job name
#SBATCH --account=OPEN-30-19                # Project account
#SBATCH --partition=qgpu                    # Partition name (qgpu for GPU jobs)
#SBATCH --gpus=3                            # Number of GPUs required
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --output=CutTreesValidation_%j.out  # Standard output log
#SBATCH --error=CutTreesValidation_%j.err   # Standard error log

# -------------------------------
# 1. Purge all loaded modules
# -------------------------------
module --force purge

# -------------------------------
# 2. Initialize Conda
# -------------------------------
# Source the conda.sh script to use conda in the script
source /scratch/project/open-26-5/jozefov_147/jozefov_147/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate phantoms_env

# -------------------------------
# 3. Set Environment Variables
# -------------------------------
export WANDB_API_KEY=${WANDB_API_KEY}

# Optional: Set CUDA_VISIBLE_DEVICES if you need to specify GPUs
# export CUDA_VISIBLE_DEVICES=0

# -------------------------------
# 4. Navigate to Project Directory
# -------------------------------
cd /scratch/project/open-26-5/jozefov_147/PhantoMS

# -------------------------------
# 5. Execute Training Script
# -------------------------------
srun --export=ALL --preserve-env python3 scripts/run_cut_trees_server.py