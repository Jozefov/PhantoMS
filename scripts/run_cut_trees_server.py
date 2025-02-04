import wandb
import yaml
import os
from datetime import datetime
import time
from typing import List, Optional

from phantoms.utils.parser import validate_config, train_model, extract_and_save_embeddings
from phantoms.optimizations.training import set_global_seeds

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
                        wandb_project_name: str):
    """
    Iterate over all configuration files and tree levels to run experiments.

    Args:
        config_dir (str): Directory containing configuration YAML files.
        experiment_parent_dir (str): Parent directory to store all experiments.
        config_files (List[str]): List of configuration YAML filenames.
        cut_tree_levels (Optional[List[int]]): List of cut_tree_at_level values.
        wandb_project_name (str): Name of the wandb project.
    """
    # Set global seeds for reproducibility
    set_global_seeds(42)

    # Create the parent experiment directory if it doesn't exist
    os.makedirs(experiment_parent_dir, exist_ok=True)

    # Iterate over each configuration file
    for config_file in config_files:
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

        # Iterate over each tree level
        for level in cut_tree_levels:
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

            # Train the model
            train_model(config, experiment_folder, config_path, level)

            # Extract and save embeddings
            extract_and_save_embeddings(config, level, experiment_folder)

            # Finish the W&B run to ensure it's properly logged
            wandb.finish()

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    # Define parameters
    config_directory = '/scratch/project/open-26-5/jozefov_147/PhantoMS/phantoms/models/retrieval/configs_server'
    experiment_parent_directory = '/scratch/project/open-26-5/jozefov_147/PhantoMS/experiments_run/full_data_cut_tress_6'

    configuration_files = [
        'config_skip_connection.yml',
        'config_skip_connection_bonus.yml',
        'config_skip_connection_dreams.yml',
        'config_skip_connection_dreams_bonus.yml'
    ]
    tree_levels = [0, 1, 2, 3]

    wandb_project_name = 'full_data_cut_tress_6'

    # Authenticate with W&B
    print("Logging into Weights & Biases...")
    wandb.login()

    # Run all experiments
    run_all_experiments(config_dir=config_directory,
                        experiment_parent_dir=experiment_parent_directory,
                        config_files=configuration_files,
                        cut_tree_levels=tree_levels,
                        wandb_project_name=wandb_project_name)