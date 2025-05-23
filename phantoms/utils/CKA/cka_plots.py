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
                                               aggregate_inter_model_cka_by_group,
                                               aggregate_alignment_metric,
                                               aggregate_projection_norm_metric,
                                               aggregate_projection_norm_by_reference)
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
    for different experiment directories, masking the diagonal (overlayed in pink),
    with improved, consistent font sizes for readability.
    """
    metric_idx = 0 if metric.lower() == 'linear' else 1

    # Compute global off-diagonal max if requested
    global_max = None
    if max_value == 'outer':
        max_vals = []
        for mat in cka_matrices.values():
            data = mat[:, :, metric_idx]
            diag_mask = np.eye(data.shape[0], dtype=bool)
            if np.any(~diag_mask):
                max_vals.append(data[~diag_mask].max())
        if max_vals:
            global_max = max(max_vals)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (exp_dir, cka_matrix) in zip(axes, cka_matrices.items()):
        data = cka_matrix[:, :, metric_idx]
        diag_mask = np.eye(data.shape[0], dtype=bool)

        # determine color limits
        if max_value == 'outer':
            vmin, vmax = 0, global_max
        else:
            offdiag = data[~diag_mask]
            vmin = 0
            vmax = offdiag.max() if offdiag.size else None

        # draw heatmap
        sns.heatmap(
            data,
            ax=ax,
            mask=diag_mask,
            cmap='Oranges',
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=True,
            cbar_kws={'shrink': 0.6},
            xticklabels=layer_names,
            yticklabels=layer_names
        )

        # invert y so layer 0 is at the bottom
        ax.invert_yaxis()

        # overlay pink on diagonal
        n = data.shape[0]
        for i in range(n):
            ax.add_patch(
                patches.Rectangle((i, i), 1, 1, facecolor='pink', edgecolor='none', zorder=3)
            )

        # styling ticks and labels (same as threshold-mask version)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        for lbl in ax.get_yticklabels():
            lbl.set_ha('right')

        # build "Max Fragmentation Stage n" title
        m = re.search(r'cut_tree_(\d+)', exp_dir)
        if m:
            stage = int(m.group(1))
            title = f"Max Fragmentation Stage {stage}"
        else:
            title = os.path.basename(exp_dir)

        ax.set_title(f"{title}\n{metric.upper()} CKA", fontsize=10)
        ax.set_xlabel("Layers", fontsize=9)
        ax.set_ylabel("Layers", fontsize=9)

    # adjust margins so labels aren’t clipped, plus supertitle
    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    plt.suptitle(f"Intra-Model {metric.upper()} CKA Heatmaps", fontsize=16)
    plt.show()

def plot_all_heatmaps_threshold_mask(cka_matrices,
                                    layer_names,
                                    metric='linear',
                                    max_value='inter',
                                    threshold=0.2):
    """
    Plot a 2x2 grid of heatmaps corresponding to the CKA matrices computed
    for different experiment directories, masking all values >= threshold
    (overlayed in pink), and titling each by its max-fragmentation stage.
    """
    metric_idx = 0 if metric.lower() == 'linear' else 1

    # compute a global max for 'outer' scaling
    global_max = None
    if max_value == 'outer':
        max_vals = []
        for mat in cka_matrices.values():
            data = mat[:, :, metric_idx]
            mask = data >= threshold
            if np.any(~mask):
                max_vals.append(data[~mask].max())
        if max_vals:
            global_max = max(max_vals)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (exp_dir, cka_matrix) in zip(axes, cka_matrices.items()):
        data = cka_matrix[:, :, metric_idx]
        mask = data >= threshold

        # determine color range
        if max_value == 'outer':
            vmin, vmax = 0, global_max
        else:
            unmasked = data[~mask]
            vmin = 0
            vmax = unmasked.max() if unmasked.size else None

        # draw the heatmap with its own colorbar
        sns.heatmap(
            data,
            ax=ax,
            mask=mask,
            cmap='Oranges',
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=True,
            cbar_kws={'shrink': 0.6},
            xticklabels=layer_names,
            yticklabels=layer_names
        )

        # flip y so layer 0 sits at the bottom
        ax.invert_yaxis()

        # overlay pink on masked cells
        n = data.shape[0]
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    ax.add_patch(
                        patches.Rectangle((j, i), 1, 1,
                                          facecolor='pink',
                                          edgecolor='none',
                                          zorder=3)
                    )

        # right-align tick labels
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        for lbl in ax.get_yticklabels():
            lbl.set_ha('right')

        # extract the fragmentation stage number from the path
        m = re.search(r'cut_tree_(\d+)', exp_dir)
        if m:
            stage = int(m.group(1))
            title = f"Max Fragmentation Stage {stage}"
        else:
            title = os.path.basename(exp_dir)

        ax.set_title(f"{title}\n{metric.upper()} CKA", fontsize=10)
        ax.set_xlabel("Layers", fontsize=9)
        ax.set_ylabel("Layers", fontsize=9)

    # give some extra room on the left for y‐labels
    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    plt.suptitle(f"Intra-Model {metric.upper()} CKA Heatmaps\n(values ≥ {threshold} masked)",
                 fontsize=16)
    plt.show()


def plot_alignment_metric_errorbars(agg_dict, layer_names, title="Average Alignment Metric vs. Layer", xlabel="Layer",
                                    ylabel="Alignment Metric"):
    """
    Plots a line chart with error bars using aggregated alignment metrics.

    Args:
        agg_dict (dict): Dictionary mapping layer name to tuple (mean, std, all_scores).
        layer_names (list): List of layer names (to preserve order on the x-axis).
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
    """
    x = np.arange(len(layer_names))
    means = []
    stds = []
    for layer in layer_names:
        mean_val, std_val, _ = agg_dict.get(layer, (np.nan, np.nan, []))
        means.append(mean_val)
        stds.append(std_val)

    plt.figure(figsize=(12, 8))
    plt.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, color='tab:blue')
    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_aggregated_alignment_by_group(experiment_groups, layer_names, top_k=5, center=True):
    """
    Given a dictionary mapping group labels (e.g., ms levels or model types) to lists of experiment
    directories, compute the aggregated alignment metric for each group and plot one curve (with error bars)
    per group.

    Args:
        experiment_groups (dict): Mapping from group label (str) to list of experiment directory paths.
        layer_names (list): List of layer names.
        top_k (int): Number of top eigenvectors to consider in the alignment metric.
        center (bool): Whether to center the data.
    """
    group_results = {}  # group label -> (x, means, stds)

    for group_label, exp_dirs in experiment_groups.items():
        agg_dict = aggregate_alignment_metric(exp_dirs, layer_names, top_k=top_k, center=center)
        means = []
        stds = []
        for layer in layer_names:
            mean_val, std_val, _ = agg_dict.get(layer, (np.nan, np.nan, []))
            means.append(mean_val)
            stds.append(std_val)
        group_results[group_label] = (np.arange(len(layer_names)), np.array(means), np.array(stds))

    plt.figure(figsize=(12, 8))
    for group_label, (x, means, stds) in group_results.items():
        plt.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, label=group_label)
    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel("Average Cosine Similarity (Top {})".format(top_k))
    plt.title("Aggregated Alignment Metric vs. Layer (by Group)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_aggregated_projection_norm_by_group(experiment_groups, layer_names, top_k=5, center=True):
    """
    Given a dictionary mapping group labels (e.g. ms levels or model types) to lists of experiment directories,
    compute the aggregated projection norm metric for each group and plot one curve (with error bars)
    per group over layers.

    Args:
        experiment_groups (dict): Mapping from group label (str) to list of experiment directory paths.
        layer_names (list): List of layer names.
        top_k (int): Number of top eigenvectors to consider.
        center (bool): Whether to center the data.
    """
    group_results = {}
    for group_label, exp_dirs in experiment_groups.items():
        agg_dict = aggregate_projection_norm_metric(exp_dirs, layer_names, top_k=top_k, center=center)
        means = []
        stds = []
        for layer in layer_names:
            mean_val, std_val, _ = agg_dict.get(layer, (np.nan, np.nan, []))
            means.append(mean_val)
            stds.append(std_val)
        group_results[group_label] = (np.arange(len(layer_names)), np.array(means), np.array(stds))

    plt.figure(figsize=(12, 8))
    for group_label, (x, means, stds) in group_results.items():
        plt.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, label=group_label)
    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel("Average Projection Norm (Top {})".format(top_k))
    plt.title("Aggregated Projection Norm vs. Layer (by Group)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_projection_norm_by_reference(exp_dirs, layer_names, top_k=5, center=True, model_label_func=None):
    """
    For a given list of experiment directories (from one experiment type or group),
    treat each experiment as the reference and compute its projection norm metric (averaged over comparisons
    with every other experiment) for each layer. Then plot one curve (with error bars) per reference.

    Args:
        exp_dirs (list): List of experiment directory paths.
        layer_names (list): List of layer names.
        top_k (int): Number of top eigenvectors to consider.
        center (bool): Whether to center the data.
        model_label_func (callable, optional): Function to extract a label from a directory.
    """
    results = aggregate_projection_norm_by_reference(exp_dirs, layer_names, top_k=top_k, center=center)

    plt.figure(figsize=(12, 8))
    x = np.arange(len(layer_names))
    for ref, (means, stds, _) in results.items():
        if model_label_func is not None:
            label = model_label_func(ref)
        else:
            label = os.path.basename(ref)
        plt.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, label=label)

    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel("Average Projection Norm (Top {})".format(top_k))
    plt.title("Projection Norm Metric vs. Layer (Reference-Based)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
