import os
import shutil
import yaml
import pandas as pd
from phantoms.utils.custom_tokenizers.BPE_tokenizers import ByteBPETokenizerWithSpecialTokens


def train_tokenizer(config, experiment_folder, config_file_path):
    # Create and save the configuration file for reference.
    os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
    config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
    shutil.copyfile(config_file_path, config_save_path)
    print(f"Configuration saved to {config_save_path}")

    print("Training tokenizer from TSV file.")
    tsv_path = config['data'].get('molecule_tsv')
    if tsv_path is None:
        raise ValueError("For tokenizer training, 'molecule_tsv' must be provided in config['data'].")

    # Read the TSV file and extract SMILES strings.
    df = pd.read_csv(tsv_path, sep="\t")
    smiles_list = df["smiles"].tolist()
    print(f"Found {len(smiles_list)} SMILES in column 'smiles'.")

    # Initialize a new Byte-level BPE tokenizer with the specified maximum length.
    max_len = config['model'].get('max_len', 200)
    tokenizer = ByteBPETokenizerWithSpecialTokens(max_len=max_len)

    # Get the save path from config.
    SMILES_TOKENIZER_SAVE_PATH = config['model'].get('smiles_tokenizer_save_path', "smiles_tokenizer.json")

    # Train the tokenizer.
    tokenizer.train(
        texts=smiles_list,
        vocab_size=config['model'].get('vocab_size', 1000),
        min_frequency=config['model'].get('min_frequency', 2),
        save_path=SMILES_TOKENIZER_SAVE_PATH,
        show_progress=True
    )
    print("Tokenizer training complete and saved to", SMILES_TOKENIZER_SAVE_PATH)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python parser.py <config_path.yml> <experiment_folder>")
        sys.exit(1)

    config_path = sys.argv[1]
    experiment_folder = sys.argv[2]

    # Load configuration from YAML.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_tokenizer(config, experiment_folder, config_path)