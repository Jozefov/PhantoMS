#!/bin/bash

LUMI_USER="jozefovf"
LUMI_SERVER="lumi"
LUMI_PATH="/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/lumi_cut_trees_DE_NOVO"
LOCAL_PATH="/Users/macbook/CODE/PhantoMS/data/de_novo_trained/cut_tree_transformer"

DIRS=(
#    "config_denovo_dreams_LUMI_cut_tree_0_2025-02-24_20-38-02"
#    "config_denovo_dreams_LUMI_cut_tree_1_2025-02-24_21-19-10"
#    "config_denovo_dreams_LUMI_cut_tree_2_2025-02-24_22-04-40"
#    "config_denovo_dreams_LUMI_cut_tree_3_2025-02-24_22-51-08"
#    "config_denovo_dreams_bonus_LUMI_cut_tree_0_2025-02-24_23-36-43"
#    "config_denovo_dreams_bonus_LUMI_cut_tree_1_2025-02-25_00-45-36"
    "config_denovo_dreams_bonus_LUMI_cut_tree_2_2025-02-25_09-38-22"
    "config_denovo_dreams_bonus_LUMI_cut_tree_3_2025-02-25_10-29-19"
#    "config_denovo_spectra_LUMI_cut_tree_0_2025-02-23_20-12-43"
#    "config_denovo_spectra_LUMI_cut_tree_1_2025-02-24_10-54-10"
#    "config_denovo_spectra_LUMI_cut_tree_2_2025-02-24_11-36-59"
#    "config_denovo_spectra_LUMI_cut_tree_3_2025-02-24_12-21-38"
#    "config_denovo_spectra_bonus_LUMI_cut_tree_0_2025-02-24_13-05-24"
#    "config_denovo_spectra_bonus_LUMI_cut_tree_1_2025-02-24_16-36-24"
#    "config_denovo_spectra_bonus_LUMI_cut_tree_2_2025-02-24_18-58-55"
#    "config_denovo_spectra_bonus_LUMI_cut_tree_3_2025-02-24_19-47-54"
)

for dir in "${DIRS[@]}"; do
    echo "Downloading embeddings from $dir..."
    mkdir -p "${LOCAL_PATH}/${dir}"
    scp -r "${LUMI_USER}@${LUMI_SERVER}:${LUMI_PATH}/${dir}/embeddings" "${LOCAL_PATH}/${dir}/"
done

echo "Download complete!"