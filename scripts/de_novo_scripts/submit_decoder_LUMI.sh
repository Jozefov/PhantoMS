#!/bin/bash
#SBATCH --job-name=DecoderTraining
#SBATCH --account=project_465001738
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=24:00:00

export WANDB_API_KEY=${WANDB_API_KEY}

cd /scratch/project_465001738/jozefov_147/PhantoMS

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONNOUSERSITE=1
source phantoms_venv/bin/activate

srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/de_novo_scripts/run_decoder_train_LUMI.py /scratch/project_465001738/jozefov_147/PhantoMS/phantoms/models/denovo/configs_server/config_decoder_LUMI.yml
