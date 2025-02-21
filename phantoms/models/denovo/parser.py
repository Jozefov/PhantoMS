# import os
# import shutil
# import yaml
# import torch
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# import pandas as pd
# import wandb
#
# from phantoms.data.datasets import MoleculeTextDataset  # our updated dataset for TSV files
# from phantoms.models.denovo.molecule_decoder import MoleculeDecoder
# from phantoms.utils.custom_tokenizers.BPE_tokenizers import ByteBPETokenizerWithSpecialTokens
# from phantoms.utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
#
#
# def train_decoder(config, experiment_folder, config_file_path):
#     # Save the configuration file for reference.
#     os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
#     config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
#     shutil.copyfile(config_file_path, config_save_path)
#     print(f"Configuration saved to {config_save_path}")
#
#     print("Training stand-alone molecule de_novo_scripts.")
#
#     # Get TSV file path (which must have a column "smiles").
#     tsv_path = config['data'].get('molecule_tsv')
#     if tsv_path is None:
#         raise ValueError("For de_novo_scripts training, 'molecule_tsv' must be provided in config['data'].")
#
#     # Load the pretrained tokenizer from the given JSON path.
#     tokenizer_path = config['model'].get('smiles_tokenizer_path')
#     if tokenizer_path is None:
#         raise ValueError("For de_novo_scripts training, 'smiles_tokenizer_path' must be provided in config['model'].")
#     tokenizer = ByteBPETokenizerWithSpecialTokens(tokenizer_path=tokenizer_path)
#
#     # Create the dataset. MoleculeTextDataset reads the TSV file and extracts the "smiles" column.
#     dataset = MoleculeTextDataset(
#         tsv_path=tsv_path,
#         tokenizer=tokenizer,
#         smiles_column="smiles",
#         max_len=config['model'].get('max_len', 200)
#     )
#
#     # Create a DataLoader with padding (using torch.nn.utils.rnn.pad_sequence).
#     pad_id = tokenizer.token_to_id(PAD_TOKEN)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=config['data'].get('batch_size', 32),
#         shuffle=True,
#         collate_fn=lambda batch: {
#             "input": torch.nn.utils.rnn.pad_sequence(
#                 [item["input"] for item in batch], batch_first=False, padding_value=pad_id
#             ),
#             "target": torch.nn.utils.rnn.pad_sequence(
#                 [item["target"] for item in batch], batch_first=False, padding_value=pad_id
#             )
#         }
#     )
#
#     # Build the de_novo_scripts model.
#     vocab_size = tokenizer.get_vocab_size()
#     model = MoleculeDecoder(
#         vocab_size=vocab_size,
#         d_model=config['model'].get('d_model', 1024),
#         nhead=config['model'].get('nhead', 4),
#         num_decoder_layers=config['model'].get('num_decoder_layers', 4),
#         dropout=config['model'].get('dropout_rate', 0.1),
#         pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
#         max_len=config['model'].get('max_len', 200)
#     )
#
#     # Initialize TensorBoard and WandB loggers.
#     tb_logger = TensorBoardLogger(
#         save_dir=os.path.join(experiment_folder, 'logs'),
#         name="tensorboard_logs"
#     )
#     wandb_logger = WandbLogger(
#         project=config['wandb']['project'],
#         entity=config['wandb']['entity'],
#         name=os.path.basename(experiment_folder),
#         log_model=False,
#         reinit=True
#     )
#
#     # Initialize the Trainer.
#     trainer = pl.Trainer(
#         max_epochs=config['trainer'].get('max_epochs', 2),
#         accelerator=config['trainer'].get('accelerator', 'cpu'),
#         devices=config['trainer'].get('devices', 1),
#         logger=[tb_logger, wandb_logger],
#         log_every_n_steps=config['trainer'].get('log_every_n_steps', 10)
#     )
#
#     # Train the model.
#     trainer.fit(model, dataloader)
#
#     # Save the pretrained de_novo_scripts weights.
#     pretrained_path = os.path.join(experiment_folder, "smiles_decoder_pretrained.pth")
#     torch.save(model.state_dict(), pretrained_path)
#     print("Decoder pretraining complete and weights saved to", pretrained_path)
#
#     wandb.finish()

#!/usr/bin/env python3
"""
Script to train the stand-alone molecule decoder on server using multiple TSV files.
This script:
  - Loads a YAML config (which can provide a list of TSVs under 'molecule_tsvs')
  - Combines the data from all TSV files (each having a "smiles" column)
  - Generates an experiment folder under
      /scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/denovo
    using the config name and a timestamp (with "decoder" in the folder name)
  - Creates the MoleculeTextDataset and DataLoader
  - Loads the pretrained tokenizer from the provided JSON path
  - Builds and trains the MoleculeDecoder model using PyTorch Lightning (GPU accelerated)
  - Logs metrics to both TensorBoard and Weights & Biases (W&B)
  - Saves the trained decoder weights

Usage:
    python run_decoder_LUMI.py <config_path.yml>
Example:
    python run_decoder_LUMI.py /scratch/project_465001738/jozefov_147/PhantoMS/phantoms/models/denovo/configs/config_decoder_server.yml
"""

import os
import sys
import time
import shutil
import yaml
import pandas as pd
import torch
import pytorch_lightning as pl
from datetime import datetime
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

# Import our dataset, model, and tokenizer modules.
from phantoms.data.datasets import MoleculeTextDataset
from phantoms.models.denovo.molecule_decoder import MoleculeDecoder
from phantoms.utils.custom_tokenizers.BPE_tokenizers import ByteBPETokenizerWithSpecialTokens
from phantoms.utils.constants import PAD_TOKEN

def train_decoder(config, experiment_folder, config_file_path):
    # Save the configuration file for reference.
    os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
    config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
    shutil.copyfile(config_file_path, config_save_path)
    print(f"Configuration saved to {config_save_path}")

    print("Training stand-alone molecule decoder.")

    # Determine TSV files: support either a single TSV or multiple TSVs.
    if 'molecule_tsvs' in config['data']:
        tsv_files = config['data']['molecule_tsvs']
    elif 'molecule_tsv' in config['data']:
        tsv_files = [config['data']['molecule_tsv']]
    else:
        raise ValueError("For decoder training, provide 'molecule_tsvs' (list) or 'molecule_tsv' in config['data'].")

    # Read and combine data from all TSV files.
    dfs = []
    for tsv in tsv_files:
        if not os.path.exists(tsv):
            print(f"WARNING: TSV file {tsv} does not exist. Skipping.")
            continue
        df = pd.read_csv(tsv, sep="\t")
        if "smiles" not in df.columns:
            print(f"WARNING: 'smiles' column not found in {tsv}. Skipping.")
            continue
        print(f"Found {len(df)} SMILES in {tsv}.")
        dfs.append(df)
    if not dfs:
        raise ValueError("No valid TSV file with a 'smiles' column found.")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total SMILES in combined dataset: {len(combined_df)}")

    # Save combined TSV to a temporary file in the experiment folder.
    temp_tsv = os.path.join(experiment_folder, "combined_molecule_data.tsv")
    combined_df.to_csv(temp_tsv, sep="\t", index=False)
    print(f"Combined TSV saved to {temp_tsv}")

    # Load the pretrained tokenizer.
    tokenizer_path = config['model'].get('smiles_tokenizer_path')
    if tokenizer_path is None:
        raise ValueError("For decoder training, 'smiles_tokenizer_path' must be provided in config['model'].")
    tokenizer = ByteBPETokenizerWithSpecialTokens(tokenizer_path=tokenizer_path)

    # Create the MoleculeTextDataset.
    dataset = MoleculeTextDataset(
        tsv_path=temp_tsv,
        tokenizer=tokenizer,
        smiles_column="smiles",
        max_len=config['model'].get('max_len', 200)
    )

    # Create a DataLoader with proper padding.
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    dataloader = DataLoader(
        dataset,
        batch_size=config['data'].get('batch_size', 32),
        shuffle=True,
        num_workers=config['data'].get('num_workers', 2),
        collate_fn=lambda batch: {
            "input": torch.nn.utils.rnn.pad_sequence(
                [item["input"] for item in batch], batch_first=False, padding_value=pad_id
            ),
            "target": torch.nn.utils.rnn.pad_sequence(
                [item["target"] for item in batch], batch_first=False, padding_value=pad_id
            )
        }
    )

    # Build the decoder model.
    vocab_size = tokenizer.get_vocab_size()
    model = MoleculeDecoder(
        vocab_size=vocab_size,
        d_model=config['model'].get('d_model', 1024),
        nhead=config['model'].get('nhead', 4),
        num_decoder_layers=config['model'].get('num_decoder_layers', 4),
        dropout=config['model'].get('dropout_rate', 0.1),
        pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
        max_len=config['model'].get('max_len', 200)
    )

    # Initialize loggers.
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(experiment_folder, 'logs'),
        name="tensorboard_logs"
    )
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=os.path.basename(experiment_folder),
        log_model=False,
        reinit=True
    )

    # Initialize the PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=config['trainer'].get('max_epochs', 10),
        accelerator=config['trainer'].get('accelerator', 'gpu'),
        devices=config['trainer'].get('devices', 8),
        logger=[tb_logger, wandb_logger],
        log_every_n_steps=config['trainer'].get('log_every_n_steps', 10),
        strategy=config['trainer'].get('strategy', 'auto')
    )

    # Start training.
    trainer.fit(model, dataloader)

    # Save the pretrained decoder weights.
    pretrained_path = os.path.join(experiment_folder, "smiles_decoder_pretrained.pth")
    torch.save(model.state_dict(), pretrained_path)
    print("Decoder pretraining complete and weights saved to", pretrained_path)

    wandb.finish()


