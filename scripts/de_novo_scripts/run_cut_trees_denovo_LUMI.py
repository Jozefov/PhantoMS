import wandb
import yaml
import os
from datetime import datetime
import time
from typing import List, Optional

from phantoms.utils.parser import validate_config, train_model, extract_and_save_embeddings
from phantoms.optimizations.training import set_global_seeds

def run_all_experiments(config_dir: str,
                        experiment_parent_dir: str,
                        config_files: List[str],
                        cut_tree_levels: Optional[List[int]],
                        wandb_project_name: str,
                        resume_config: Optional[str] = None,
                        resume_level: Optional[int] = None):
    """
    Iterate over all configuration files and tree levels to run experiments.
    When resume_config and resume_level are provided, the function will skip
    over the experiments until it reaches the specified configuration file and
    tree level, and then continue from there.

    Args:
        config_dir (str): Directory containing configuration YAML files.
        experiment_parent_dir (str): Parent directory to store all experiments.
        config_files (List[str]): List of configuration YAML filenames.
        cut_tree_levels (Optional[List[int]]): List of cut_tree_at_level values.
        wandb_project_name (str): Name of the wandb project.
        resume_config (Optional[str]): Configuration file to resume from.
        resume_level (Optional[int]): Tree level within resume_config to resume from.
    """

    # Set global seeds for reproducibility
    set_global_seeds(42)

    # Create the parent experiment directory if it doesn't exist
    os.makedirs(experiment_parent_dir, exist_ok=True)

    # If resume_config is not provided, we start normally (i.e. resume_found=True)
    resume_found = resume_config is None

    for config_file in config_files:
        # If we are in resume mode, skip config files until we find the resume_config
        if not resume_found:
            if config_file == resume_config:
                resume_found = True
                # For the resume config, only run levels >= resume_level
                current_levels = [level for level in cut_tree_levels if level >= resume_level]
                print(f"Resuming {config_file} starting from tree level {resume_level}")
            else:
                print(f"Skipping {config_file} (waiting to resume from {resume_config})")
                continue
        else:
            # For the config file we are resuming from, filter levels; for the rest, run all levels.
            if resume_config is not None and config_file == resume_config:
                current_levels = [level for level in cut_tree_levels if level >= resume_level]
            else:
                current_levels = cut_tree_levels

        config_path = os.path.join(config_dir, config_file)

        if not os.path.exists(config_path):
            print(f"Configuration file {config_path} does not exist. Skipping.")
            continue

        # Load the configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate the configuration
        try:
            validate_config(config)
        except ValueError as ve:
            print(f"Configuration validation error in {config_file}: {ve}")
            continue

        # Iterate over each tree level for this configuration file
        for level in current_levels:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Adding a slight sleep to ensure different timestamps
            time.sleep(1.0)

            # Define a unique experiment folder name
            config_name = os.path.splitext(os.path.basename(config_file))[0]
            experiment_folder_name = f"{config_name}_cut_tree_{level}_{timestamp}"
            experiment_folder = os.path.join(experiment_parent_dir, experiment_folder_name)

            # Print experiment details for debugging
            print(f"\nRunning Experiment: {experiment_folder_name}")
            print(f"W&B Project: {wandb_project_name}")
            print(f"Cut Tree Level: {level}")

            # Create the experiment folder
            os.makedirs(experiment_folder, exist_ok=True)

            # Modify the config dict's 'trainer.checkpoint_dir' to point to experiment_folder subdirectories
            config['trainer']['checkpoint_dir'] = os.path.join(experiment_folder, 'checkpoints')

            # Optionally, modify 'experiment_base_name' to include the experiment_folder name or set to a unified value
            config['experiment_base_name'] = 'experiment_trial'  # Or any desired base name

            # Update W&B project name
            config['wandb']['project'] = wandb_project_name

            # Train the model with error handling in case it fails.
            try:
                train_model(config, experiment_folder, config_path, level)
            except Exception as e:
                print(f"Error during training for {config_file} at level {level}: {e}")
                # Optionally, log the error, save the state, or even break if desired.
                continue

            # Extract and save embeddings
            try:
                extract_and_save_embeddings(config, level, experiment_folder)
            except Exception as e:
                print(f"Error during embedding extraction for {config_file} at level {level}: {e}")

            # Finish the W&B run to ensure it's properly logged
            wandb.finish()

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":

    # Define parameters
    config_directory = '/scratch/project_465001738/jozefov_147/PhantoMS/phantoms/models/denovo/configs_server'
    experiment_parent_directory = '/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/lumi_cut_trees_DE_NOVO'

    configuration_files = [
        'config_denovo_spectra_LUMI.yml',
        'config_denovo_spectra_bonus_LUMI.yml',
        'config_denovo_dreams_LUMI.yml',
        'config_denovo_dreams_bonus_LUMI.yml'
    ]
    tree_levels = [0, 1, 2, 3]

    wandb_project_name = 'lumi_cut_trees_DE_NOVO'

    # Optional resume parameters:
    # To resume from a failure at, for example, 'config_skip_connection_dreams_LUMI.yml' and tree level 2,
    # set resume_config and resume_level accordingly.
    # resume_config = 'config_skip_connection_dreams_LUMI.yml'
    # resume_level = 3
    resume_config = None
    resume_level = None

    # Authenticate with W&B
    print("Logging into Weights & Biases...")
    wandb.login()

    # Run all experiments (with resume capability)
    run_all_experiments(config_dir=config_directory,
                        experiment_parent_dir=experiment_parent_directory,
                        config_files=configuration_files,
                        cut_tree_levels=tree_levels,
                        wandb_project_name=wandb_project_name,
                        resume_config=resume_config,
                        resume_level=resume_level)