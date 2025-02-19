#!/bin/bash


LUMI_USER="jozefovf"
LUMI_SERVER="lumi"
LUMI_PATH="/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/lumi_cut_trees_MSE"
LOCAL_PATH="/Users/macbook/CODE/PhantoMS/data/retrieval_trained/cut_tree_MSE"

DIRS=(
    "config_skip_connection_LUMI_cut_tree_0_2025-02-18_19-48-58"
    "config_skip_connection_LUMI_cut_tree_1_2025-02-18_20-27-21"
    "config_skip_connection_LUMI_cut_tree_2_2025-02-18_21-08-04"
    "config_skip_connection_LUMI_cut_tree_3_2025-02-18_21-49-39"
    "config_skip_connection_bonus_LUMI_cut_tree_0_2025-02-18_22-29-53"
    "config_skip_connection_bonus_LUMI_cut_tree_1_2025-02-18_23-15-20"
    "config_skip_connection_bonus_LUMI_cut_tree_2_2025-02-19_00-01-32"
    "config_skip_connection_bonus_LUMI_cut_tree_3_2025-02-19_00-47-31"
    "config_skip_connection_dreams_LUMI_cut_tree_0_2025-02-19_01-33-16"
    "config_skip_connection_dreams_LUMI_cut_tree_1_2025-02-19_02-12-27"
    "config_skip_connection_dreams_LUMI_cut_tree_2_2025-02-19_02-54-27"
    "config_skip_connection_dreams_LUMI_cut_tree_3_2025-02-19_03-35-19"
)


for dir in "${DIRS[@]}"; do
    echo "Downloading embeddings from $dir..."
    scp -r "${LUMI_USER}@${LUMI_SERVER}:${LUMI_PATH}/${dir}/embeddings" "${LOCAL_PATH}/${dir}/"
done

echo "Download complete!"