import os
import shutil
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pandas as pd
import wandb

from phantoms.data.datasets import MoleculeTextDataset  # our updated dataset for TSV files
from phantoms.models.denovo.molecule_decoder import MoleculeDecoder
from phantoms.utils.custom_tokenizers.BPE_tokenizers import ByteBPETokenizerWithSpecialTokens
from phantoms.utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def train_decoder(config, experiment_folder, config_file_path):
    # Save the configuration file for reference.
    os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
    config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
    shutil.copyfile(config_file_path, config_save_path)
    print(f"Configuration saved to {config_save_path}")

    print("Training stand-alone molecule decoder.")

    # Get TSV file path (which must have a column "smiles").
    tsv_path = config['data'].get('molecule_tsv')
    if tsv_path is None:
        raise ValueError("For decoder training, 'molecule_tsv' must be provided in config['data'].")

    # Load the pretrained tokenizer from the given JSON path.
    tokenizer_path = config['model'].get('smiles_tokenizer_path')
    if tokenizer_path is None:
        raise ValueError("For decoder training, 'smiles_tokenizer_path' must be provided in config['model'].")
    tokenizer = ByteBPETokenizerWithSpecialTokens(tokenizer_path=tokenizer_path)

    # Create the dataset. MoleculeTextDataset reads the TSV file and extracts the "smiles" column.
    dataset = MoleculeTextDataset(
        tsv_path=tsv_path,
        tokenizer=tokenizer,
        smiles_column="smiles",
        max_len=config['model'].get('max_len', 200)
    )

    # Create a DataLoader with padding (using torch.nn.utils.rnn.pad_sequence).
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    dataloader = DataLoader(
        dataset,
        batch_size=config['data'].get('batch_size', 32),
        shuffle=True,
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

    # Initialize TensorBoard and WandB loggers.
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

    # Initialize the Trainer.
    trainer = pl.Trainer(
        max_epochs=config['trainer'].get('max_epochs', 2),
        accelerator=config['trainer'].get('accelerator', 'cpu'),
        devices=config['trainer'].get('devices', 1),
        logger=[tb_logger, wandb_logger],
        log_every_n_steps=config['trainer'].get('log_every_n_steps', 10)
    )

    # Train the model.
    trainer.fit(model, dataloader)

    # Save the pretrained decoder weights.
    pretrained_path = os.path.join(experiment_folder, "smiles_decoder_pretrained.pth")
    torch.save(model.state_dict(), pretrained_path)
    print("Decoder pretraining complete and weights saved to", pretrained_path)

    wandb.finish()