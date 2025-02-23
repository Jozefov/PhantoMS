import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
import re
import matplotlib.patches as patches


from phantoms.utils.CKA.cka_processing import (compute_sorted_eigenvalues,
                                               compute_effective_rank,
                                               compute_cum_explained_variance,
                                               aggregate_cum_explained_variance_for_type,
                                               aggregate_effective_rank_for_type,
                                               aggregate_inter_model_cka_by_group)
from phantoms.utils.CKA.cka_data import load_embeddings


def plot_effective_rank_vs_layer_multi(experiment_dirs, layer_names, model_label_func=None):
    """
    For a list of experiment directories (each corresponding to one model),
    load the embeddings for the given layers, compute the effective rank for each layer,
    and plot effective rank vs. layer for all models in one plot.

    Args:
        experiment_dirs (list): List of experiment directory paths.
        layer_names (list): List of layer names (order is preserved).
        model_label_func (callable, optional): A function that takes an experiment directory
            string and returns a label for that model. If None, the basename of the directory is used.
    """
    effective_ranks_dict = {}

    for exp_dir in experiment_dirs:
        embeddings = load_embeddings(exp_dir, layer_names)
        effective_ranks = []
        for layer in layer_names:
            if layer in embeddings:
                # embeddings[layer] is assumed to have shape [num_examples, representation_dim]
                sorted_eigvals = compute_sorted_eigenvalues(embeddings[layer])
                eff_rank = compute_effective_rank(sorted_eigvals)
                effective_ranks.append(eff_rank)
            else:
                effective_ranks.append(np.nan)

        # Generate a label for this model
        if model_label_func is not None:
            label = model_label_func(exp_dir)
        else:
            # Default: use the basename of the directory
            label = os.path.basename(exp_dir)
        effective_ranks_dict[label] = effective_ranks

    # Plot effective rank vs. layer for all models on one plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(layer_names))
    for label, eff_ranks in effective_ranks_dict.items():
        plt.plot(x, eff_ranks, marker='o', label=label)

    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel("Effective Rank")
    plt.title("Effective Rank vs. Layer Across Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_avg_cum_explained_variance_multi(experiment_dirs, layer_names, max_components=50, center=True,
                                          model_label_func=None):
    """
    For each experiment directory (model), loads the embeddings for the given layers,
    computes the cumulative explained variance curve for each layer, averages these curves
    across layers, and then plots one curve per experiment.

    Args:
        experiment_dirs (list): List of experiment directory paths.
        layer_names (list): List of layer names (order is preserved).
        max_components (int): Number of components to include in the plot.
        center (bool): If True, center the data when computing covariance.
        model_label_func (callable, optional): Function that takes an experiment directory
            string and returns a label for that model. If None, the basename is used.
    """
    avg_cumvar_dict = {}

    for exp_dir in tqdm(experiment_dirs, desc="Processing experiments"):

        embeddings = load_embeddings(exp_dir, layer_names)
        cumvar_curves = []

        for layer in layer_names:
            if layer in embeddings:
                emb = embeddings[layer]  # shape: [num_examples, representation_dim]
                cum_explained = compute_cum_explained_variance(emb, center=center, max_components=max_components)
                cumvar_curves.append(cum_explained)
            else:
                continue

        if cumvar_curves:
            # Average the cumulative curves across layers (pointwise)
            avg_curve = np.mean(np.stack(cumvar_curves, axis=0), axis=0)
        else:
            avg_curve = np.zeros(max_components)

        if model_label_func is not None:
            label = model_label_func(exp_dir)
        else:
            label = os.path.basename(exp_dir)
        avg_cumvar_dict[label] = avg_curve

    plt.figure(figsize=(12, 8))
    x = np.arange(1, max_components + 1)
    for label, avg_curve in avg_cumvar_dict.items():
        plt.plot(x, avg_curve, marker='o', label=label)

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Average Cumulative Explained Variance Across Layers\nfor Each Experiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_aggregated_metrics_by_type(experiment_types, layer_names, max_components=100, center=True,
                                    metric="effective_rank"):
    """
    Given a dictionary mapping model type labels to lists of experiment directories,
    aggregates the chosen metric (effective rank or cumulative explained variance) over experiments
    for each model type and plots one curve per model type.

    Args:
        experiment_types (dict): Mapping from model type (str) to a list of experiment directories.
        layer_names (list): List of layer names.
        max_components (int): Number of components (only used for cumulative explained variance).
        center (bool): Whether to center the data.
        metric (str): "effective_rank" or "cum_explained".
    """
    agg_metric_dict = {}

    for model_type, exp_dirs in experiment_types.items():
        if metric == "effective_rank":
            avg_metric = aggregate_effective_rank_for_type(exp_dirs, layer_names, center=center)
            # For effective rank, x-axis will be layer indices
            x = np.arange(len(layer_names))
            agg_metric_dict[model_type] = (x, avg_metric)
        elif metric == "cum_explained":
            avg_metric = aggregate_cum_explained_variance_for_type(exp_dirs, layer_names, max_components=max_components,
                                                                   center=center)
            # For cumulative explained variance, x-axis is number of components.
            x = np.arange(1, max_components + 1)
            agg_metric_dict[model_type] = (x, avg_metric)

    plt.figure(figsize=(12, 8))
    for model_type, (x, y) in agg_metric_dict.items():
        plt.plot(x, y, marker='o', label=model_type)

    if metric == "effective_rank":
        plt.xlabel("Layer Index")
        plt.ylabel("Average Effective Rank")
        plt.title("Aggregated Effective Rank vs. Layer (by Model Type)")
        plt.xticks(x, layer_names, rotation=45, ha='right')
    elif metric == "cum_explained":
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Aggregated Cumulative Explained Variance (by Model Type)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_inter_model_cka(layer_similarities, metric='linear'):
    """
    Plots a line plot where the x-axis is the layer index (or name) and the y-axis is the
    average inter-model CKA similarity for that layer (with error bars).

    Args:
        layer_similarities (dict): Mapping from layer name to list of CKA scores.
        metric (str): 'linear' or 'rbf' indicating the type of CKA used.
    """
    layers = list(layer_similarities.keys())
    means = []
    stds = []
    for layer in layers:
        scores = np.array(layer_similarities[layer])
        means.append(np.mean(scores))
        stds.append(np.std(scores))

    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(layers)), means, yerr=stds, fmt='-o', capsize=5, color='tab:blue')
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel(f"Average {metric.upper()} CKA")
    plt.title(f"Inter-Model {metric.upper()} CKA Across Corresponding Layers")
    plt.tight_layout()
    plt.show()


def plot_aggregated_inter_model_cka_by_group(experiment_groups, layer_names, metric='linear'):
    """
    Plots the aggregated inter-model CKA curves for each group.

    Args:
        experiment_groups (dict): Mapping from group label (e.g. "dreams") to a list of
                                  experiment directory paths.
        layer_names (list): List of layer names.
        metric (str): 'linear' or 'rbf' indicating which CKA metric is used.
    """
    group_results = aggregate_inter_model_cka_by_group(experiment_groups, layer_names, metric=metric)

    plt.figure(figsize=(12, 8))
    x = np.arange(len(layer_names))
    for group_label, (means, stds) in group_results.items():
        plt.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, label=group_label)

    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel(f"Average {metric.upper()} CKA")
    plt.title(f"Aggregated Inter-Model {metric.upper()} CKA Across Layers (by Group)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_heatmaps(cka_matrices, layer_names, metric='linear', max_value='inter'):
    """
    Plot a 2x2 grid of heatmaps corresponding to the CKA matrices computed
    for different experiment directories.

    Args:
        cka_matrices (dict): Dictionary with keys as experiment directory paths
                             and values as CKA matrices (shape: [num_layers, num_layers, 2]).
        layer_names (list): List of layer names (used for x- and y-axis labels).
        metric (str): 'linear' for linear CKA or 'rbf' for RBF kernel CKA.
        max_value (str): 'inter' to scale each heatmap independently by its maximum
                         off-diagonal value, or 'outer' to use the maximum off-diagonal
                         value across all heatmaps for a common scale.
    """

    metric_idx = 0 if metric.lower() == 'linear' else 1

    # If using outer scaling, compute the global maximum (off-diagonal) across all experiments.
    global_max = None
    if max_value == 'outer':
        max_vals = []
        for cka_matrix in cka_matrices.values():
            data = cka_matrix[:, :, metric_idx]
            mask = np.eye(data.shape[0], dtype=bool)
            if np.sum(~mask) > 0:
                max_vals.append(np.max(data[~mask]))
        if max_vals:
            global_max = np.max(max_vals)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (exp_dir, cka_matrix) in zip(axes, cka_matrices.items()):
        data = cka_matrix[:, :, metric_idx]
        mask = np.eye(data.shape[0], dtype=bool)

        # Determine vmax: if 'outer', use global_max; else compute for this heatmap.
        if max_value == 'outer':
            vmax = global_max
        else:
            # Compute maximum from off-diagonals for this heatmap.
            if np.sum(~mask) > 0:
                vmax = np.max(data[~mask])
            else:
                vmax = None

        # sns.heatmap(data, ax=ax, annot=False, fmt=".2f",
        #             xticklabels=layer_names, yticklabels=layer_names,
        #             cmap='coolwarm', mask=mask, vmin=0, vmax=vmax)


        # Plot the heatmap with the diagonal masked.
        sns.heatmap(data, ax=ax, annot=False, fmt=".2f",
                    xticklabels=layer_names, yticklabels=layer_names,
                    cmap='Oranges', mask=mask, vmin=0, vmax=vmax)

        # Overlay pink patches on the diagonal cells.
        # Each cell in the heatmap is 1x1 in the coordinate system.
        for i in range(len(layer_names)):
            ax.add_patch(patches.Rectangle((i, i), 1, 1, fill=True, color='pink', lw=0, zorder=3))

        match = re.search(r'cut_tree_\d+', exp_dir)
        if match:
            title = match.group(0)
        else:
            title = os.path.basename(exp_dir)

        ax.set_title(f"{title}\n{metric.upper()} CKA", fontsize=10)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Layers")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

def plot_all_heatmaps_threshold_mask(cka_matrices, layer_names, metric='linear', max_value='inter', threshold=0.2):
    """
    Plot a 2x2 grid of heatmaps corresponding to the CKA matrices computed
    for different experiment directories, masking all values greater than or equal
    to a given threshold. Masked cells are overlaid with pink.

    Args:
        cka_matrices (dict): Dictionary with keys as experiment directory paths
                             and values as CKA matrices (shape: [num_layers, num_layers, 2]).
        layer_names (list): List of layer names (used for x- and y-axis labels).
        metric (str): 'linear' for linear CKA or 'rbf' for RBF kernel CKA.
        max_value (str): 'inter' to scale each heatmap independently by its maximum
                         unmasked value, or 'outer' to use the maximum unmasked value
                         across all heatmaps for a common scale.
        threshold (float): All values in the CKA matrices that are >= threshold will be masked.
    """

    metric_idx = 0 if metric.lower() == 'linear' else 1

    # If using outer scaling, compute the global maximum from unmasked values across experiments.
    global_max = None
    if max_value == 'outer':
        max_vals = []
        for cka_matrix in cka_matrices.values():
            data = cka_matrix[:, :, metric_idx]
            # Mask cells where the value is >= threshold.
            mask = data >= threshold
            if np.sum(~mask) > 0:
                max_vals.append(np.max(data[~mask]))
        if max_vals:
            global_max = np.max(max_vals)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (exp_dir, cka_matrix) in zip(axes, cka_matrices.items()):
        data = cka_matrix[:, :, metric_idx]
        # Create a mask for all cells where value is >= threshold.
        mask = data >= threshold

        # Determine vmax: if using outer scaling, use global_max; otherwise compute local maximum from unmasked cells.
        if max_value == 'outer':
            vmax = global_max
        else:
            if np.sum(~mask) > 0:
                vmax = np.max(data[~mask])
            else:
                vmax = None

        # Plot heatmap with cells >= threshold masked.
        sns.heatmap(data, ax=ax, annot=False, fmt=".2f",
                    xticklabels=layer_names, yticklabels=layer_names,
                    cmap='Oranges', mask=mask, vmin=0, vmax=vmax)

        # Overlay pink patches on each masked cell.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=True, color='pink', lw=0, zorder=3))

        match = re.search(r'cut_tree_\d+', exp_dir)
        if match:
            title = match.group(0)
        else:
            title = os.path.basename(exp_dir)

        ax.set_title(f"{title}\n{metric.upper()} CKA", fontsize=10)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Layers")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle(f"Intra-Model {metric.upper()} CKA Heatmaps (Values >= {threshold} Masked)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()