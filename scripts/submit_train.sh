#!/bin/bash

# Init logging dir and common file tracking submissions
train_dir="/scratch/project/open-26-5/MSCLIP"
outdir="${train_dir}/submissions"
mkdir -p "${outdir}"
outfile="${outdir}/submissions.csv"

# Generate random key for a job
job_key=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10 ; echo '')
job_dir="${outdir}/${job_key}"
mkdir -p "${job_dir}"

# Copy config files into the job directory
config_main="${job_dir}/configs/config_main.yaml"
config_model="${job_dir}/configs/config_clip.yaml"
mkdir -p "${job_dir}/configs"
cp "${train_dir}/configs/config_main.yaml" "${config_main}"
cp "${train_dir}/configs/config_clip.yaml" "${config_model}"

# Submit
job_id=$(sbatch \
  --account=OPEN-30-19 \
  --partition=qgpu \
  --gpus=8 \
  --nodes=1 \
  --time=24:00:00 \
  --output="${job_dir}/stdout.txt" \
  --error="${job_dir}/errout.txt" \
  --job-name="${job_key}" \
  --export=job_dir="${job_dir}",config_main="${config_main}",config_model="${config_model}" \
  "${train_dir}/train.sh"
)

# Log
submission="$(date),${job_id},${job_key}"
echo "${submission}" >> "${outfile}"
echo "${submission}"