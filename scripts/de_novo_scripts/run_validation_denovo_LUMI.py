#!/usr/bin/env python3
import os
import re
import yaml

from pytorch_lightning.loggers import WandbLogger
from phantoms.models.denovo.parser import predict_denovo

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Hardcoded list of experiment directories (relative to base_dir)
    base_dir = "/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/lumi_cut_trees_DE_NOVO"
    experiments = [
        # "config_denovo_dreams_LUMI_cut_tree_0_2025-02-24_20-38-02",
        # "config_denovo_dreams_LUMI_cut_tree_1_2025-02-24_21-19-10",
        # "config_denovo_dreams_LUMI_cut_tree_2_2025-02-24_22-04-40",
        # "config_denovo_dreams_LUMI_cut_tree_3_2025-02-24_22-51-08",
        # "config_denovo_dreams_bonus_LUMI_cut_tree_0_2025-02-24_23-36-43",
        # "config_denovo_dreams_bonus_LUMI_cut_tree_1_2025-02-25_00-45-36",
        # "config_denovo_dreams_bonus_LUMI_cut_tree_2_2025-02-25_09-38-22",
        # "config_denovo_dreams_bonus_LUMI_cut_tree_3_2025-02-25_10-29-19",
        # "config_denovo_spectra_LUMI_cut_tree_0_2025-02-23_20-12-43",
        "config_denovo_spectra_LUMI_cut_tree_1_2025-02-24_10-54-10",
        "config_denovo_spectra_LUMI_cut_tree_2_2025-02-24_11-36-59",
        "config_denovo_spectra_LUMI_cut_tree_3_2025-02-24_12-21-38",
        "config_denovo_spectra_bonus_LUMI_cut_tree_0_2025-02-24_13-05-24",
        "config_denovo_spectra_bonus_LUMI_cut_tree_1_2025-02-24_16-36-24",
        "config_denovo_spectra_bonus_LUMI_cut_tree_2_2025-02-24_18-58-55",
        "config_denovo_spectra_bonus_LUMI_cut_tree_3_2025-02-24_19-47-54"
    ]

    for exp in experiments:
        exp_folder = os.path.join(base_dir, exp)
        config_folder = os.path.join(exp_folder, "configs")
        config_files = [f for f in os.listdir(config_folder) if f.endswith(".yml")]
        if not config_files:
            print(f"No config file found in {config_folder}; skipping {exp_folder}")
            continue
        config_path = os.path.join(config_folder, config_files[0])
        config = load_config(config_path)

        # Extract cut_tree_level from the experiment folder name (expects 'cut_tree_X')
        match = re.search(r"cut_tree_(\d+)", exp)
        if match:
            cut_tree_level = int(match.group(1))
        else:
            print(f"Could not extract cut_tree_level from folder name: {exp}; skipping.")
            continue

        # Set checkpoint path in config (assumes checkpoints/final_model.ckpt exists in the folder)
        checkpoint_path = os.path.join(exp_folder, "checkpoints", "final_model.ckpt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path} for experiment {exp_folder}; skipping.")
            continue
        config["model"]["checkpoint_path"] = checkpoint_path

        # Create a WandbLogger for this run.
        # All runs will be under the project "lumi_cut_trees_validation_DE_NOVO" and the run name is the experiment folder name.
        wandb_logger = WandbLogger(
            project="lumi_cut_trees_validation_DE_NOVO",
            name=exp,
            save_dir=exp_folder
        )

        print(f"\nRunning prediction for experiment: {exp_folder} (cut_tree_level: {cut_tree_level})")
        predict_denovo(config, exp_folder, cut_tree_level, logger=wandb_logger)

        # Finish the wandb run so that the next run starts a new one.
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()