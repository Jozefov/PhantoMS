import os
import shutil
import yaml
import random
import numpy as np
import wandb
import torch
import pytorch_lightning as pl
from datetime import datetime
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from phantoms.callbacks.freeze_decoder import FreezeDecoderCallback
from massspecgym.data.data_module import MassSpecDataModule
from massspecgym.featurize import SpectrumFeaturizer
from massspecgym.data.datasets import MSnRetrievalDataset, MSnDataset
from massspecgym.data.transforms import MolFingerprinter
from phantoms.utils.custom_tokenizers.BPE_tokenizers import ByteBPETokenizerWithSpecialTokens

from phantoms.models.retrieval.gnn_retrieval_model_skip_connection import GNNRetrievalSkipConnections
from phantoms.models.denovo.GATDeNovoTransformer import GATDeNovoTransformer
from phantoms.models.denovo.molecule_decoder import MoleculeDecoder
from phantoms.data.datasets import MoleculeTextDataset

from phantoms.utils.data import save_embeddings
from phantoms.optimizations.training import set_global_seeds


def validate_config(config):
    required_keys = ['featurizer', 'data', 'model', 'metrics', 'optimizer', 'trainer', 'experiment_base_name', 'wandb']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")


def train_model(config, experiment_folder, config_file_path, cut_tree_level=None):
    print(f"\nStarting training for {experiment_folder}")
    set_global_seeds(config.get("seed", 42))


    task = config.get("task", "retrieval")

    featurizer = SpectrumFeaturizer(config['featurizer'], mode='torch')

    if task == "retrieval":
        print("Using retrieval task/model.")
        spectra_mgf = config['data'].get('spectra_mgf')
        candidates_json = config['data'].get('candidates_json')
        cache_pth = config['data'].get('cache_path')
        dataset = MSnRetrievalDataset(
            pth=spectra_mgf,
            candidates_pth=candidates_json,
            cache_pth=cache_pth,
            featurizer=featurizer,
            mol_transform=MolFingerprinter(fp_size=config['model']['fp_size']),
            cut_tree_at_level=cut_tree_level,
            max_allowed_deviation=config['data']['max_allowed_deviation'],
            hierarchical_tree=config['data']['hierarchical_tree']
        )
        model = GNNRetrievalSkipConnections(
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

    elif task == "denovo":
        print("Using de novo task/model.")
        spectra_mgf = config['data'].get('spectra_mgf')
        dataset = MSnDataset(
            pth=spectra_mgf,
            featurizer=featurizer,
            mol_transform=None,
            cut_tree_at_level=cut_tree_level,
            max_allowed_deviation=config['data']['max_allowed_deviation'],
            hierarchical_tree=config['data']['hierarchical_tree']
        )
        tokenizer_path = config['model'].get('smiles_tokenizer_path')
        if tokenizer_path is None:
            raise ValueError("For de novo task, 'smiles_tokenizer_path' must be provided in config['model'].")
        smiles_tokenizer = ByteBPETokenizerWithSpecialTokens(tokenizer_path=tokenizer_path)
        model = GATDeNovoTransformer(
            input_dim=config['model']['node_feature_dim'],
            d_model=config['model'].get('d_model', 1024),
            nhead=config['model'].get('nhead', 8),
            num_gat_layers=config['model'].get('num_gat_layers', 3),
            num_decoder_layers=config['model'].get('num_decoder_layers', 6),
            num_gat_heads=config['model'].get('num_gat_heads', 8),
            gat_dropout=config['model'].get('gat_dropout', 0.6),
            smiles_tokenizer=smiles_tokenizer,
            dropout=config['model'].get('dropout_rate', 0.1),
            max_smiles_len=config['model'].get('max_smiles_len', 200),
            k_predictions=config['model'].get('k_predictions', 1),
            temperature=config['model'].get('temperature', 1.0),
            pre_norm=config['model'].get('pre_norm', False),
            use_formula=config['model'].get('use_formula', False),
            formula_embedding_dim=config['model'].get('formula_embedding_dim', 64),
            beam_width=config['model'].get('beam_width', 1),
            at_ks=config['metrics']['at_ks'],
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        if config['model'].get('load_pretrained_decoder', False):
            pretrained_state = torch.load(config['model'].get('decoder_pretrained_path'), map_location="cpu")
            model.load_pretrained_decoder(pretrained_state)
            print("Loaded pretrained de_novo_scripts into de novo model.")
    else:
        raise ValueError(f"Unknown task: {task}")

    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=config['data']['batch_size'],
        split_pth=config['data']['split_file'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )

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

    checkpoint_dir = os.path.join(experiment_folder, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config['trainer']['checkpoint_monitor'],
            save_top_k=config['trainer']['save_top_k'],
            mode=config['trainer']['checkpoint_mode'],
            dirpath=checkpoint_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step')
    ]
    if task == "denovo" and config['model'].get('freeze_decoder', False):
        callbacks.append(FreezeDecoderCallback(freeze_epochs=config['model'].get('freeze_epochs', 3)))

    trainer = pl.Trainer(
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        num_nodes=config['trainer'].get('num_nodes', 1),
        strategy=config['trainer'].get('strategy', 'auto'),
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=config['trainer'].get('check_val_every_n_epoch', 1),
        logger=[tb_logger, wandb_logger],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        limit_train_batches=config['trainer']['limit_train_batches'],
        limit_val_batches=config['trainer']['limit_val_batches'],
        limit_test_batches=config['trainer']['limit_test_batches'],
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=data_module)

    if hasattr(model, "beam_width"):
        orig_beam = model.beam_width
        model.beam_width = config['model'].get('test_beam_width', 10)
    trainer.test(model, datamodule=data_module)
    if hasattr(model, "beam_width"):
        model.beam_width = orig_beam

    model_save_path = os.path.join(checkpoint_dir, 'final_model.ckpt')
    trainer.save_checkpoint(model_save_path)
    print(f"Model saved to {model_save_path}")

    wandb.finish()

    os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
    config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
    shutil.copyfile(config_file_path, config_save_path)
    print(f"Configuration saved to {config_save_path}")

def extract_and_save_embeddings(config, cut_tree_level, experiment_folder):
    print(f"\nExtracting embeddings for {experiment_folder}")
    task = config.get("task", "retrieval")
    checkpoint_dir = os.path.join(experiment_folder, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'final_model.ckpt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Skipping embedding extraction.")
        return
    if task == "retrieval":
        from phantoms.models.retrieval.gnn_retrieval_model_skip_connection import GNNRetrievalSkipConnections
        model = GNNRetrievalSkipConnections.load_from_checkpoint(
            checkpoint_path,
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
    elif task == "denovo":
        from phantoms.models.denovo.GATDeNovoTransformer import GATDeNovoTransformer
        tokenizer_path = config['model'].get('smiles_tokenizer_path')
        smiles_tokenizer = ByteBPETokenizerWithSpecialTokens(tokenizer_path=tokenizer_path)
        model = GATDeNovoTransformer.load_from_checkpoint(
            checkpoint_path,
            input_dim=config['model']['node_feature_dim'],
            d_model=config['model'].get('d_model', 1024),
            nhead=config['model'].get('nhead', 8),
            num_gat_layers=config['model'].get('num_gat_layers', 3),
            num_decoder_layers=config['model'].get('num_decoder_layers', 6),
            num_gat_heads=config['model'].get('num_gat_heads', 8),
            gat_dropout=config['model'].get('gat_dropout', 0.6),
            smiles_tokenizer=smiles_tokenizer,
            dropout=config['model'].get('dropout_rate', 0.1),
            max_smiles_len=config['model'].get('max_smiles_len', 200),
            k_predictions=config['model'].get('k_predictions', 1),
            temperature=config['model'].get('temperature', 1.0),
            pre_norm=config['model'].get('pre_norm', False),
            use_formula=config['model'].get('use_formula', False),
            formula_embedding_dim=config['model'].get('formula_embedding_dim', 64),
            beam_width=config['model'].get('beam_width', 1),
            at_ks=config['metrics']['at_ks'],
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    model.eval()

    if task == "retrieval":
        from massspecgym.data.datasets import MSnRetrievalDataset
        dataset = MSnRetrievalDataset(
            pth=config['data']['spectra_mgf'],
            candidates_pth=config['data'].get('candidates_json'),
            cache_pth=config['data'].get('cache_pth'),
            featurizer=SpectrumFeaturizer(config['featurizer'], mode='torch'),
            mol_transform=MolFingerprinter(fp_size=config['model']['fp_size']),
            cut_tree_at_level=cut_tree_level,
            max_allowed_deviation=config['data']['max_allowed_deviation'],
            hierarchical_tree=config['data']['hierarchical_tree']
        )
    elif task == "denovo":
        from massspecgym.data.datasets import MSnDataset
        dataset = MSnDataset(
            pth=config['data']['spectra_mgf'],
            featurizer=SpectrumFeaturizer(config['featurizer'], mode='torch'),
            mol_transform=None,
            cut_tree_at_level=cut_tree_level,
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

    embeddings_save_dir = os.path.join(experiment_folder, 'embeddings')
    os.makedirs(embeddings_save_dir, exist_ok=True)

    test_loader = data_module.test_dataloader()
    save_embeddings(model, test_loader, embeddings_save_dir)
    print(f"Embeddings saved to {embeddings_save_dir}")