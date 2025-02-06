#!/bin/bash
#SBATCH --job-name=CutTreesValidation       # Job name
#SBATCH --account=project_465001738         # LUMI project account
#SBATCH --partition=standard-g              # GPU partition on LUMI
#SBATCH --gpus-per-node=8                   # Request 2 GPUs per node
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --time=48:00:00                     # Time limit (hrs:min:sec)
#SBATCH --output=CutTreesValidation_%j.out  # Standard output log
#SBATCH --error=CutTreesValidation_%j.err   # Standard error log

# -------------------------------
# 1. Purge all loaded modules
# -------------------------------
module --force purge

# -------------------------------
# 2. Load necessary modules
# -------------------------------
# On LUMI it is recommended to start with CrayEnv to initialize the Cray programming environment.
module load CrayEnv
# (Optional: If your workflow requires a specific LUMI stack version, load it here.
# For example: module load LUMI/24.03 partition/C)

# -------------------------------
# 3. Initialize and activate Conda
# -------------------------------
# Source the conda initialization script.
source /scratch/project_465001738/jozefov_147/miniconda3/etc/profile.d/conda.sh
# Activate your Conda environment (named phantoms_env)
conda activate phantoms_env

# -------------------------------
# 4. Ensure WANDB API key is available
# -------------------------------
# (Assuming your ~/.bashrc already exports WANDB_API_KEY; we export it here again for safety)
export WANDB_API_KEY=${WANDB_API_KEY}

# -------------------------------
# 5. Change to the project directory
# -------------------------------
cd /scratch/project_465001738/jozefov_147/PhantoMS

# -------------------------------
# 6. Execute the training script with srun
# -------------------------------
# The --export=ALL --preserve-env options ensure that environment variables (including WANDB_API_KEY)
# are passed to the job step.
srun --export=ALL --preserve-env python3 scripts/run_cut_trees_server.py