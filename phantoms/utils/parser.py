import wandb
import os
import shutil

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from phantoms.models.retrieval.gnn_retrieval_model_skip_connection import GNNRetrievalSkipConnections
from massspecgym.data.data_module import MassSpecDataModule
from massspecgym.data.transforms import MolFingerprinter
from massspecgym.data.datasets import MSnRetrievalDataset
from massspecgym.featurize import SpectrumFeaturizer
import pytorch_lightning as pl
from phantoms.utils.data import save_embeddings

def validate_config(config):
    """
    Validate the configuration dictionary.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        ValueError: If any required key is missing.
    """
    required_keys = ['featurizer', 'data', 'model', 'metrics', 'optimizer', 'trainer', 'experiment_base_name', 'wandb']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")


def train_model(config, cut_tree_level, experiment_folder, config_file_path):
    """
    Train a single model based on the provided configuration and tree level.

    Args:
        config (dict): Configuration dictionary.
        cut_tree_level (int): The tree depth level for the experiment.
        experiment_folder (str): Path to the experiment's unique folder.
        config_file_path (str): Path to the original config file for copying.
    """
    print(f"\nStarting training for {experiment_folder}")

    # Initialize Featurizer
    featurizer = SpectrumFeaturizer(config['featurizer'], mode='torch')

    spectra_mgf = config['data'].get('spectra_mgf')
    candidates_json = config['data'].get('candidates_json')

    # Initialize Dataset with specific cut_tree_at_level
    dataset_msn = MSnRetrievalDataset(
        pth=spectra_mgf,
        candidates_pth=candidates_json,
        featurizer=featurizer,
        mol_transform=MolFingerprinter(fp_size=config['model']['fp_size']),
        cut_tree_at_level=cut_tree_level,
        max_allowed_deviation=config['data']['max_allowed_deviation']
    )

    data_module_msn = MassSpecDataModule(
        dataset=dataset_msn,
        batch_size=config['data']['batch_size'],
        split_pth=config['data']['split_file'],
        num_workers=config['data']['num_workers'],
    )

    # Initialize Model
    model = GNNRetrievalSkipConnections(
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['fp_size'],
        node_feature_dim=config['model']['node_feature_dim'],
        dropout_rate=config['model'].get('dropout_rate', 0.2),
        bottleneck_factor=config['model'].get('bottleneck_factor', 1.0),
        num_skipblocks=config['model'].get('num_skipblocks', 3),
        num_gcn_layers=config['model'].get('num_gcn_layers', 3),
        use_formula=config['model'].get('use_formula', False),
        formula_embedding_dim=config['model'].get('formula_embedding_dim', 64),
        at_ks=config['metrics']['at_ks'],
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    # Initialize Loggers with experiment_name
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(experiment_folder, 'logs'),
        name="tensorboard_logs",
        version=""
    )

    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=os.path.basename(experiment_folder),
        log_model=False,
        reinit=True
    )

    # Define experiment-specific checkpoint directory
    checkpoint_dir = os.path.join(experiment_folder, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = pl.Trainer(
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        num_nodes=config['trainer'].get('num_nodes', 1),
        strategy='ddp' if config['trainer']['accelerator'] == 'ddp' else None,
        max_epochs=config['trainer']['max_epochs'],
        logger=[tb_logger, wandb_logger],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        limit_train_batches=config['trainer']['limit_train_batches'],
        limit_val_batches=config['trainer']['limit_val_batches'],
        limit_test_batches=config['trainer']['limit_test_batches'],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor=config['trainer']['checkpoint_monitor'],
                save_top_k=config['trainer']['save_top_k'],
                mode=config['trainer']['checkpoint_mode'],
                dirpath=checkpoint_dir,
                filename='gnn_retrieval-{epoch:02d}-{val_loss:.2f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    # Train the model
    trainer.fit(model, datamodule=data_module_msn)

    # Test the model
    trainer.test(model, datamodule=data_module_msn)

    # Save the trained model
    model_save_path = os.path.join(checkpoint_dir, 'final_model.ckpt')
    trainer.save_checkpoint(model_save_path)
    print(f"Model saved to {model_save_path}")

    wandb.finish()

    # Save the config file to the experiment folder for reference
    os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
    config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
    shutil.copyfile(config_file_path, config_save_path)
    print(f"Configuration saved to {config_save_path}")


def extract_and_save_embeddings(config, cut_tree_level, experiment_folder):
    """
    Extract and save embeddings from a trained model based on the provided configuration and tree level.

    Args:
        config (dict): Configuration dictionary.
        cut_tree_level (int): The tree depth level for the experiment.
        experiment_folder (str): Path to the experiment's unique folder.
    """
    print(f"\nExtracting embeddings for {experiment_folder}")

    # Load the trained model
    checkpoint_dir = os.path.join(experiment_folder, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'final_model.ckpt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for {experiment_folder} at {checkpoint_path}. Skipping embedding extraction.")
        return

    model = GNNRetrievalSkipConnections.load_from_checkpoint(
        checkpoint_path,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['fp_size'],
        node_feature_dim=config['model']['node_feature_dim'],
        dropout_rate=config['model'].get('dropout_rate', 0.2),
        bottleneck_factor=config['model'].get('bottleneck_factor', 1.0),
        num_skipblocks=config['model'].get('num_skipblocks', 3),
        num_gcn_layers=config['model'].get('num_gcn_layers', 3),
        use_formula=config['model'].get('use_formula', False),
        formula_embedding_dim=config['model'].get('formula_embedding_dim', 64),
        at_ks=config['metrics']['at_ks'],
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )
    model.eval()

    # Initialize Featurizer
    featurizer = SpectrumFeaturizer(config['featurizer'], mode='torch')

    spectra_mgf = config['data'].get('spectra_mgf')
    candidates_json = config['data'].get('candidates_json')

    # Initialize Dataset with specific cut_tree_at_level
    dataset_msn = MSnRetrievalDataset(
        pth=spectra_mgf,
        candidates_pth=candidates_json,
        featurizer=featurizer,
        mol_transform=MolFingerprinter(fp_size=config['model']['fp_size']),
        cut_tree_at_level=cut_tree_level,
        max_allowed_deviation=config['data']['max_allowed_deviation']
    )

    # Initialize DataModule with fixed shuffle seed
    data_module_msn = MassSpecDataModule(
        dataset=dataset_msn,
        batch_size=config['data']['batch_size'],
        split_pth=config['data']['split_file'],
        num_workers=config['data']['num_workers'],
    )

    # Prepare and setup the test data
    data_module_msn.prepare_data()
    data_module_msn.setup("test")

    # Define embedding save directory
    embeddings_save_dir = os.path.join(experiment_folder, 'embeddings')
    os.makedirs(embeddings_save_dir, exist_ok=True)

    # Use the test dataloader for embeddings extraction
    test_loader = data_module_msn.test_dataloader()

    # Extract and save embeddings
    save_embeddings(model, test_loader, embeddings_save_dir)
    print(f"Embeddings saved to {embeddings_save_dir}")