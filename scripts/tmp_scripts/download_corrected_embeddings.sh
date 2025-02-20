#!/bin/bash


LUMI_USER="jozefovf"
LUMI_SERVER="lumi"
LUMI_PATH="/scratch/project_465001738/jozefov_147/PhantoMS/experiments_run/lumi_cut_trees_COSINE"
LOCAL_PATH="/Users/macbook/CODE/PhantoMS/data/retrieval_trained/cut_tree_COSINE"


DIRS=(
    "config_skip_connection_LUMI_cut_tree_0_2025-02-19_09-00-57"
    "config_skip_connection_LUMI_cut_tree_1_2025-02-19_09-38-36"
    "config_skip_connection_LUMI_cut_tree_2_2025-02-19_10-20-04"
    "config_skip_connection_LUMI_cut_tree_3_2025-02-19_11-02-28"
    "config_skip_connection_bonus_LUMI_cut_tree_0_2025-02-19_11-44-59"
    "config_skip_connection_bonus_LUMI_cut_tree_1_2025-02-19_12-31-04"
    "config_skip_connection_bonus_LUMI_cut_tree_2_2025-02-19_13-16-56"
    "config_skip_connection_bonus_LUMI_cut_tree_3_2025-02-19_14-04-53"
    "config_skip_connection_dreams_LUMI_cut_tree_0_2025-02-19_14-51-13"
    "config_skip_connection_dreams_LUMI_cut_tree_1_2025-02-19_15-31-19"
    "config_skip_connection_dreams_LUMI_cut_tree_2_2025-02-19_16-12-30"
    "config_skip_connection_dreams_LUMI_cut_tree_3_2025-02-19_16-56-31"
    "config_skip_connection_dreams_bonus_LUMI_cut_tree_0_2025-02-19_17-38-07"
    "config_skip_connection_dreams_bonus_LUMI_cut_tree_1_2025-02-19_18-24-19"
    "config_skip_connection_dreams_bonus_LUMI_cut_tree_2_2025-02-19_19-10-33"
    "config_skip_connection_dreams_bonus_LUMI_cut_tree_3_2025-02-19_19-57-08"
)


for dir in "${DIRS[@]}"; do
    echo "Downloading corrected_embeddings from $dir..."
    scp -r "${LUMI_USER}@${LUMI_SERVER}:${LUMI_PATH}/${dir}/corrected_embeddings" "${LOCAL_PATH}/${dir}/"
done

echo "Download complete!"