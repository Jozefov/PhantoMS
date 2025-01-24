#!/bin/bash
#SBATCH --job-name=CutTreesValidation       # Job name
#SBATCH --account=OPEN-30-19                # Project account
#SBATCH --partition=qgpu                    # Partition name (qgpu for GPU jobs)
#SBATCH --gpus=1                            # Number of GPUs required
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --time=1:00:00                      # Time limit hrs:min:sec
#SBATCH --output=CutTreesValidation_%j.out  # Standard output log
#SBATCH --error=CutTreesValidation_%j.err   # Standard error log

# -------------------------------
# 1. Load Necessary Modules
# -------------------------------
ml purge                                 # Remove all loaded modules
ml Anaconda3/2021.05                     # Load Anaconda module
ml CUDA/11.3                             # Load CUDA module (adjust version if needed)

# -------------------------------
# 2. Activate Conda Environment
# -------------------------------
source activate phantoms_env              # Activate your Conda environment

# -------------------------------
# 3. Set Environment Variables
# -------------------------------
export WANDB_API_KEY=your_actual_api_key_here  # Replace with your actual W&B API key

# -------------------------------
# 4. Navigate to Project Directory
# -------------------------------
cd /scratch/project/open-26-5/jozefov_147/PhantoMS  # Change to your project directory

# -------------------------------
# 5. Execute Training Script
# -------------------------------
srun --export=ALL,WANDB_API_KEY=$WANDB_API_KEY --preserve-env python3 scripts/run_cut_trees_server.py