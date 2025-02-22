import os
import shutil
import yaml
import pandas as pd
from phantoms.utils.custom_tokenizers.BPE_tokenizers import ByteBPETokenizerWithSpecialTokens


def train_tokenizer(config, experiment_folder, config_file_path):
    os.makedirs(os.path.join(experiment_folder, 'configs'), exist_ok=True)
    config_save_path = os.path.join(experiment_folder, 'configs', os.path.basename(config_file_path))
    shutil.copyfile(config_file_path, config_save_path)
    print(f"Configuration saved to {config_save_path}")

    print("Training tokenizer from TSV file(s).")

    # Check if a list of TSV files is provided, otherwise fallback to a single TSV.
    tsv_files = None
    if 'molecule_tsvs' in config['data']:
        tsv_files = config['data']['molecule_tsvs']
    elif 'molecule_tsv' in config['data']:
        tsv_files = [config['data']['molecule_tsv']]
    else:
        raise ValueError(
            "For tokenizer training, provide 'molecule_tsvs' (list of paths) or 'molecule_tsv' in config['data'].")

    smiles_list = []
    for tsv_path in tsv_files:
        if not os.path.exists(tsv_path):
            print(f"WARNING: TSV file {tsv_path} does not exist. Skipping.")
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        if "smiles" not in df.columns:
            print(f"WARNING: 'smiles' column not found in {tsv_path}. Skipping.")
            continue
        curr_smiles = df["smiles"].tolist()
        print(f"Found {len(curr_smiles)} SMILES in {tsv_path}.")
        smiles_list.extend(curr_smiles)

    print(f"Total SMILES for training: {len(smiles_list)}")

    max_len = config['model'].get('max_len', 200)
    tokenizer = ByteBPETokenizerWithSpecialTokens(max_len=max_len)

    SMILES_TOKENIZER_SAVE_PATH = config['model'].get('smiles_tokenizer_save_path', "smiles_tokenizer.json")

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

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_tokenizer(config, experiment_folder, config_path)