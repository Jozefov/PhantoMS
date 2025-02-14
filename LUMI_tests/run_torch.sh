#!/bin/bash -l
#SBATCH --job-name="TorchTest"
#SBATCH --output=torchtest.out
#SBATCH --error=torchtest.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00
#SBATCH --account=project_465001738

# Clean environment
module purge
# Load the appropriate LUMI software stack
module load LUMI/24.03
# Load the PyTorch container module (adjust the version string as needed)
module load LUMI PyTorch/2.4.1-rocm-6.1.3-python-3.12-singularity-20241007

echo "Container image: $SIFPYTORCH"

# Run a simple test inside the container
srun singularity exec $SIFPYTORCH python3 -c "import torch; print('Torch version:', torch.__version__); print('GPU count:', torch.cuda.device_count())"