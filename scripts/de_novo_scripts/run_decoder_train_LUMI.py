#!/usr/bin/env python3
import os
import sys
import time
import yaml
from datetime import datetime
from phantoms.models.denovo.parser import train_decoder

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_decoder_train_LUMI.py <config_path.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create a unique experiment folder under the "denovo" parent directory.
    base_exp_dir = "/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/denovo"
    os.makedirs(base_exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time.sleep(1.0)  # Ensure a unique timestamp
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    experiment_folder_name = f"{config_name}_decoder_{timestamp}"
    experiment_folder = os.path.join(base_exp_dir, experiment_folder_name)
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"Experiment folder: {experiment_folder}")

    # Call the train_decoder function (from phantoms.models.denovo.parser)
    train_decoder(config, experiment_folder, config_path)

if __name__ == "__main__":
    main()