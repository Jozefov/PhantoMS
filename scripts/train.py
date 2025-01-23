import os
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path

import utils
from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
from models.clip import MSCLIP
from models.dreams_fp import DreaMSFP

# Set HF_HOME location for downloading datasets from huggingface
os.environ["HF_HOME"] = "/scratch/project/open-26-5/hf_data/"

def main(config):

    # TODO: add dropout arg. What about dropout in DreaMS?
    # TODO: model checkpointing callback
    # TODO: config paths to argparse
    # TODO: experiment with fp16 training to have larger batch size

    # Seed everything
    pl.seed_everything(config["seed"])

    # Init dataset
    dataset = RetrievalDataset(
        spec_transform=SpecTokenizer(n_peaks=config["n_peaks"]),
        mol_transform=MolFingerprinter(),
    )

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # Init model
    if config["model_name"] == "MSCLIP":
        model = MSCLIP(config)
    elif config["model_name"] == "DreaMSFP":
        model = DreaMSFP(config)
    else:
        raise ValueError(f"Model {config['model_name']} not found")

    # Init wandb logger
    logger = None
    if not config["no_wandb"]:
        logger = pl.loggers.WandbLogger(
            name=config["wandb_name"],
            project="MSCLIP",
            log_model=False,
        )

    # Init callbacks for checkpointing and early stopping
    callbacks = []
    for i, monitor in enumerate(model.get_checkpoint_monitors()):
        monitor_name = monitor["monitor"]
        # checkpoint = pl.callbacks.ModelCheckpoint(
        #     monitor=monitor_name,
        #     save_top_k=1,
        #     mode=monitor["mode"],
        #     dirpath=Path(args.project_name) / args.job_key,
        #     filename=f'{{step:06d}}-{{{monitor_name}:03.03f}}',
        #     auto_insert_metric_name=True,
        #     save_last=(i == 0)
        # )
        # callbacks.append(checkpoint)

        if monitor.get("early_stopping", False):
            early_stopping = pl.callbacks.EarlyStopping(
                monitor=monitor_name,
                mode=monitor["mode"],
                verbose=True,
                patience=config["early_stopping_patience"]
            )
            callbacks.append(early_stopping)

    # Init trainer
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=device,
        strategy="ddp",
        devices=config["devices"],
        max_epochs=config["max_epochs"],
        logger=logger,
        log_every_n_steps=config["log_every_n_steps"],
        val_check_interval=config["val_check_interval"],
        callbacks=callbacks,
    )

    # Validate before training
    if config["validate_before_training"]:
        data_module.prepare_data()
        data_module.setup()
        trainer.validate(model, datamodule=data_module)

    # Train
    trainer.fit(model, datamodule=data_module)

    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--config_main", type=Path, required=True)
    parser.add_argument("--config_model", type=Path, required=True)
    args = parser.parse_args()

    # Load configs
    main_config_pth = Path(args.config_main)
    model_config_pth = Path(args.config_model)
    config = OmegaConf.merge(
        OmegaConf.load(main_config_pth),  # Main config
        OmegaConf.load(model_config_pth),  # Model config
        OmegaConf.create(vars(args))  # Command line args
    )
    config["job_key"] = args.run_dir.name

    # Run script
    main(config)
