#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate msclip

# Configure paths
export HF_HOME=/scratch/project/open-26-5/hf_data/
work_dir=/scratch/project/open-26-5/MSCLIP
cd $work_dir

# Check if job_dir and config paths are set
if [ -z "${job_dir}" ] || [ -z "${config_main}" ] || [ -z "${config_model}" ]; then
    export job_dir="${work_dir}/submissions/interactive_$(date +%Y%m%d_%H%M%S)"
    export config_main="${work_dir}/configs/config_main.yaml"
    export config_model="${work_dir}/configs/config_clip.yaml"
    echo "Environment variables job_dir, config_main, or config_model are not set."
    echo "Using default values:"
    echo "  job_dir: ${job_dir}"
    echo "  config_main: ${config_main}" 
    echo "  config_model: ${config_model}"
fi

# Ensure job_dir exists
mkdir -p "${job_dir}"

# Run python training script
srun --export=ALL --preserve-env python3 train.py \
# python3 train.py \
    --run_dir $job_dir \
    --config_main "${config_main}" \
    --config_model "${config_model}"