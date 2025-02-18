#!/bin/bash
##SBATCH --job-name=CutTreesValidation       # Job name
##SBATCH --account=project_465001738         # LUMI project account
##SBATCH --partition=standard-g              # GPU partition on LUMI
##SBATCH --gpus-per-node=1                   # Request 2 GPUs per node
##SBATCH --nodes=1                           # Number of nodes
##SBATCH --time=4:00:00                     # Time limit (hrs:min:sec)
##SBATCH --output=CutTreesValidation_%j.out  # Standard output log
##SBATCH --error=CutTreesValidation_%j.err   # Standard error log
#
#
#
## -------------------------------
## 1. Purge all loaded modules
## -------------------------------
#module --force purge
#
## -------------------------------
## 2. Load necessary modules
## -------------------------------
## On LUMI it is recommended to start with CrayEnv to initialize the Cray programming environment.
#module load CrayEnv
## (Optional: If your workflow requires a specific LUMI stack version, load it here.
## For example: module load LUMI/24.03 partition/C)
#
## -------------------------------
## 3. Initialize and activate Conda
## -------------------------------
## Source the conda initialization script.
#source /scratch/project_465001738/jozefov_147/miniconda3/etc/profile.d/conda.sh
## Activate your Conda environment (named phantoms_env)
#conda activate phantoms_env
#
## -------------------------------
## 4. Set GPU environment variables for distributed training
## -------------------------------
##export SLURM_GPUS_PER_NODE=8
##export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
## -------------------------------
## 5. Ensure WANDB API key is available
## -------------------------------
## (Assuming your ~/.bashrc already exports WANDB_API_KEY; we export it here again for safety)
#export WANDB_API_KEY=${WANDB_API_KEY}
#
## -------------------------------
## 6. Change to the project directory
## -------------------------------
#cd /scratch/project_465001738/jozefov_147/PhantoMS
#
## -------------------------------
## 6. Execute the training script with srun
## -------------------------------
## The --export=ALL --preserve-env options ensure that environment variables (including WANDB_API_KEY)
## are passed to the job step.
#srun --export=ALL --preserve-env python3 scripts/run_cut_trees_server.py


#SBATCH --account=project_465001738
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=48:00:00

# Load necessary modules and activate your environment

export WANDB_API_KEY=${WANDB_API_KEY}
cd /scratch/project_465001738/jozefov_147/PhantoMS

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONNOUSERSITE=1
source phantoms_venv/bin/activate

#export SLURM_GPUS_PER_NODE=2
#export CUDA_VISIBLE_DEVICES=0,1

# (Optional) Print some info for debugging
#echo "Running on node $(hostname)"
#echo "CUDA_VISIBLE_DEVICES before binding: $CUDA_VISIBLE_DEVICES"

# Launch your Python script using srun.
# This will create 2 separate tasks, each with its own SLURM_LOCALID.
srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/run_cut_trees_LUMI.py