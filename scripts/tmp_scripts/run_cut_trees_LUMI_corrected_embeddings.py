
"""
Script to extract corrected embeddings from a specified list of experiment folders.
For each folder (e.g. "config_skip_connection_LUMI_cut_tree_0_2025-02-19_09-00-57"),
this script:
  - Parses the tree depth from the folder name.
  - Loads the YAML config file from the folder’s "configs" subdirectory.
  - Loads the trained model checkpoint ("checkpoints/final_model.ckpt").
  - Constructs the test dataset (using the proper tree depth for cut_tree_at_level).
  - Extracts new embeddings using the updated model (which collects pre-ReLU activations).
  - Saves the new embeddings in a new subfolder called "corrected_embeddings".

"""

import os
import re
import yaml


from massspecgym.featurize import SpectrumFeaturizer
from massspecgym.data.datasets import MSnRetrievalDataset
from massspecgym.data.transforms import MolFingerprinter
from massspecgym.data.data_module import MassSpecDataModule
from phantoms.models.retrieval.gnn_retrieval_model_skip_connection import GNNRetrievalSkipConnections
from phantoms.utils.data import save_embeddings

def process_experiment_folder(exp_folder: str):
    """
    Process one experiment folder: load its config and checkpoint,
    re-create the test dataset (with proper tree depth), extract embeddings
    using the corrected (pre-ReLU) version of the model, and save them.
    """
    m = re.search(r'cut_tree_(\d+)', exp_folder)
    if not m:
        print(f"WARNING: Could not determine tree depth from folder name {exp_folder}. Skipping.")
        return
    tree_depth = int(m.group(1))
    print(f"\n=== Processing folder: {exp_folder} (tree depth: {tree_depth}) ===")


    config_dir = os.path.join(exp_folder, "configs")
    if not os.path.exists(config_dir):
        print(f"WARNING: No 'configs' folder found in {exp_folder}. Skipping.")
        return

    config_files = [f for f in os.listdir(config_dir) if f.endswith(('.yml', '.yaml'))]
    if not config_files:
        print(f"WARNING: No YAML config file found in {config_dir}. Skipping.")
        return

    config_path = os.path.join(config_dir, config_files[0])
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    print(f"Loaded config: {config_path}")

    ckpt_path = os.path.join(exp_folder, "checkpoints", "final_model.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint file not found at {ckpt_path}. Skipping folder {exp_folder}.")
        return

    print(f"Loading model checkpoint from: {ckpt_path}")
    model = GNNRetrievalSkipConnections.load_from_checkpoint(
        ckpt_path,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['fp_size'],
        node_feature_dim=config['model']['node_feature_dim'],
        dropout_rate=config['model'].get('dropout_rate', 0.2),
        bottleneck_factor=config['model'].get('bottleneck_factor', 1.0),
        num_skipblocks=config['model'].get('num_skipblocks', 6),
        num_gcn_layers=config['model'].get('num_gcn_layers', 3),
        use_formula=config['model'].get('use_formula', False),
        formula_embedding_dim=config['model'].get('formula_embedding_dim', 64),
        gnn_layer_type=config['model'].get('gnn_layer_type', 'GCNConv'),
        nheads=config['model'].get('nheads', 0),
        at_ks=config['metrics']['at_ks'],
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )
    model.eval()

    featurizer = SpectrumFeaturizer(config['featurizer'], mode='torch')
    dataset = MSnRetrievalDataset(
        pth=config['data']['spectra_mgf'],
        candidates_pth=config['data'].get('candidates_json'),
        cache_pth=config['data'].get('cache_pth'),
        featurizer=featurizer,
        mol_transform=MolFingerprinter(fp_size=config['model']['fp_size']),
        cut_tree_at_level=tree_depth,
        max_allowed_deviation=config['data']['max_allowed_deviation'],
        hierarchical_tree=config['data']['hierarchical_tree']
    )

    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=config['data']['batch_size'],
        split_pth=config['data']['split_file'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    data_module.prepare_data()
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    corrected_emb_dir = os.path.join(exp_folder, "corrected_embeddings")
    os.makedirs(corrected_emb_dir, exist_ok=True)

    print("Extracting corrected embeddings (using pre-ReLU activations)...")
    save_embeddings(model, test_loader, corrected_emb_dir)
    print(f"Saved corrected embeddings to: {corrected_emb_dir}")

def main():
    # Parent directory
    parent_dir = "/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/lumi_cut_trees_COSINE"

    # List of target experiment folder names:
    target_folder_names = [
        "config_skip_connection_LUMI_cut_tree_0_2025-02-19_09-00-57",
        "config_skip_connection_LUMI_cut_tree_1_2025-02-19_09-38-36",
        "config_skip_connection_LUMI_cut_tree_2_2025-02-19_10-20-04",
        "config_skip_connection_LUMI_cut_tree_3_2025-02-19_11-02-28",
        "config_skip_connection_bonus_LUMI_cut_tree_0_2025-02-19_11-44-59",
        "config_skip_connection_bonus_LUMI_cut_tree_1_2025-02-19_12-31-04",
        "config_skip_connection_bonus_LUMI_cut_tree_2_2025-02-19_13-16-56",
        "config_skip_connection_bonus_LUMI_cut_tree_3_2025-02-19_14-04-53",
        "config_skip_connection_dreams_LUMI_cut_tree_0_2025-02-19_14-51-13",
        "config_skip_connection_dreams_LUMI_cut_tree_1_2025-02-19_15-31-19",
        "config_skip_connection_dreams_LUMI_cut_tree_2_2025-02-19_16-12-30",
        "config_skip_connection_dreams_LUMI_cut_tree_3_2025-02-19_16-56-31",
        "config_skip_connection_dreams_bonus_LUMI_cut_tree_0_2025-02-19_17-38-07",
        "config_skip_connection_dreams_bonus_LUMI_cut_tree_1_2025-02-19_18-24-19",
        "config_skip_connection_dreams_bonus_LUMI_cut_tree_2_2025-02-19_19-10-33",
        "config_skip_connection_dreams_bonus_LUMI_cut_tree_3_2025-02-19_19-57-08"
    ]

    target_folders = [os.path.join(parent_dir, name) for name in target_folder_names]

    # Process each target folder.
    for exp_folder in target_folders:
        if os.path.isdir(exp_folder):
            try:
                process_experiment_folder(exp_folder)
            except Exception as e:
                print(f"ERROR processing folder {exp_folder}: {e}")
        else:
            print(f"WARNING: Folder {exp_folder} does not exist.")

    print("\nAll specified experiment folders processed.")

if __name__ == "__main__":
    main()