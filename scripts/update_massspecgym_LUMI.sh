#!/bin/bash
source /scratch/project_465001738/jozefov_147/PhantoMS/phantoms_venv/bin/activate

echo "Using Python from: $(which python)"
echo "Using pip from: $(which pip)"
echo "Current environment: $VIRTUAL_ENV"

echo "Removing massspecgym"
pip uninstall massspecgym -y

echo "Updating massspecgym from GitHub..."
pip install --upgrade git+ssh://git@github.com/Jozefov/MassSpecGymMSn.git@main

if [ $? -eq 0 ]; then
    echo "massspecgym has been updated successfully."
else
    echo "Failed to update massspecgym."
    exit 1
fi