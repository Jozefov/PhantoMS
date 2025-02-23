import numpy as np
from phantoms.utils.CKA.cka_data import load_embeddings
from phantoms.utils.CKA.CKA import linear_CKA, kernel_CKA
import itertools
from tqdm import tqdm

def compute_sorted_eigenvalues(embeddings, center=True):
    """
    Computes the eigenvalues of the covariance matrix of the embeddings.

    Args:
        embeddings (np.ndarray): Array of shape [num_examples, representation_dim].
        center (bool): If True, subtract the mean of each column.

    Returns:
        sorted_eigvals (np.ndarray): Eigenvalues sorted in descending order.
    """
    # Center the embeddings if required.
    if center:
        X_centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    else:
        X_centered = embeddings

    # Compute the covariance matrix. Using rowvar=False means each column is a variable.
    cov_matrix = np.cov(X_centered, rowvar=False)

    # For a symmetric covariance matrix, use eigh (which returns real eigenvalues)
    eigvals, _ = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues in descending order
    sorted_eigvals = np.sort(eigvals)[::-1]
    return sorted_eigvals


def compute_effective_rank(sorted_eigvals):
    """
    Computes the effective rank of the representation based on the eigenvalue spectrum.

    Args:
        sorted_eigvals (np.ndarray): Sorted eigenvalues (descending order).

    Returns:
        effective_rank (float): The effective rank of the representation.
    """
    eps = 1e-12
    total = np.sum(sorted_eigvals) + eps
    p = sorted_eigvals / total
    entropy = -np.sum(p * np.log(p + eps))
    effective_rank = np.exp(entropy)
    return effective_rank


def shared_subspace_analysis(X, Y, center=True):
    """
    Performs shared subspace analysis by projecting the eigenvectors of the covariance
    matrix of X onto the similarity matrix of Y.

    Args:
        X (np.ndarray): Activation matrix from model A of shape [num_examples, dim].
        Y (np.ndarray): Activation matrix from model B of shape [num_examples, dim].
        center (bool): If True, center the data by subtracting the mean.

    Returns:
        eigvals_sorted (np.ndarray): Sorted eigenvalues of S_X in descending order.
        projection_norms (np.ndarray): Norms of the projected eigenvectors from Y Y^T.
        cosines (np.ndarray): Cosine similarities between each eigenvector and its projection.
    """
    if center:
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
    else:
        X_centered = X
        Y_centered = Y

    # Compute similarity (covariance) matrices:
    S_X = np.dot(X_centered, X_centered.T)
    S_Y = np.dot(Y_centered, Y_centered.T)

    # Eigen-decomposition of S_X (since it is symmetric)
    eigvals, U = np.linalg.eigh(S_X)

    # Sort eigenvalues (and eigenvectors) in descending order:
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    U_sorted = U[:, sorted_indices]

    projection_norms = []
    cosines = []

    # For each eigenvector u_i, compute the projection via S_Y:
    for i in range(U_sorted.shape[1]):
        u = U_sorted[:, i]
        v = np.dot(S_Y, u)
        norm_v = np.linalg.norm(v)
        projection_norms.append(norm_v)
        if norm_v < 1e-12:
            cos_sim = 0
        else:
            cos_sim = np.dot(u, v) / norm_v
        cosines.append(cos_sim)

    return eigvals_sorted, np.array(projection_norms), np.array(cosines)

def compute_cum_explained_variance(embeddings, center=True, max_components=None):
    """
    Computes the cumulative explained variance curve for a given layer's embeddings.

    Args:
        embeddings (np.ndarray): Array of shape [num_examples, representation_dim].
        center (bool): If True, subtract the mean from each column.
        max_components (int, optional): If provided, limits the number of components shown.

    Returns:
        cum_explained (np.ndarray): 1D array of cumulative explained variance.
    """
    # Center the data if requested
    if center:
        X_centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    else:
        X_centered = embeddings

    # Compute covariance matrix (each column is a variable)
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues using eigh (covariance matrix is symmetric)
    eigvals, _ = np.linalg.eigh(cov_matrix)
    sorted_eigvals = np.sort(eigvals)[::-1]  # descending order

    if max_components is not None:
        sorted_eigvals = sorted_eigvals[:max_components]

    cum_explained = np.cumsum(sorted_eigvals) / np.sum(sorted_eigvals)
    return cum_explained


def aggregate_effective_rank_for_type(exp_dirs, layer_names, center=True):
    """
    For a list of experiment directories (all for a given model type),
    compute the effective rank for each layer and average across experiments.

    Args:
        exp_dirs (list): List of experiment directory paths.
        layer_names (list): List of layer names.
        center (bool): Whether to center the embeddings.

    Returns:
        avg_eff_rank (np.ndarray): 1D array (length equal to number of layers) with the averaged effective rank.
    """
    eff_rank_list = []
    for exp_dir in exp_dirs:
        embeddings = load_embeddings(exp_dir, layer_names)
        eff_ranks = []
        for layer in layer_names:
            if layer in embeddings:
                sorted_eigvals = compute_sorted_eigenvalues(embeddings[layer], center=center)
                eff_rank = compute_effective_rank(sorted_eigvals)
                eff_ranks.append(eff_rank)
            else:
                eff_ranks.append(np.nan)
        eff_rank_list.append(np.array(eff_ranks))
    eff_rank_array = np.stack(eff_rank_list, axis=0)
    avg_eff_rank = np.nanmean(eff_rank_array, axis=0)
    return avg_eff_rank


def aggregate_cum_explained_variance_for_type(exp_dirs, layer_names, max_components=100, center=True):
    """
    For a list of experiment directories (all for a given model type),
    compute the cumulative explained variance curve for each layer, average these curves
    over layers for each experiment, and then average across experiments.

    Args:
        exp_dirs (list): List of experiment directory paths.
        layer_names (list): List of layer names.
        max_components (int): Number of components to consider.
        center (bool): Whether to center the data.

    Returns:
        avg_curve (np.ndarray): 1D array (length = max_components) with the averaged cumulative explained variance curve.
    """
    curve_list = []
    for exp_dir in exp_dirs:
        embeddings = load_embeddings(exp_dir, layer_names)
        layer_curves = []
        for layer in layer_names:
            if layer in embeddings:
                emb = embeddings[layer]
                cum_explained = compute_cum_explained_variance(emb, center=center, max_components=max_components)
                layer_curves.append(cum_explained)
        if layer_curves:
            layer_curves = np.stack(layer_curves, axis=0)
            avg_curve_exp = np.mean(layer_curves, axis=0)
            curve_list.append(avg_curve_exp)
    if curve_list:
        curve_array = np.stack(curve_list, axis=0)
        avg_curve = np.mean(curve_array, axis=0)
    else:
        avg_curve = np.zeros(max_components)
    return avg_curve

def compute_layer_cka(embeddings_model1, embeddings_model2, layer_name1, layer_name2):
    """
    Compute CKA scores between two sets of embeddings for a specific layer.
    """
    X = embeddings_model1.get(layer_name1)
    Y = embeddings_model2.get(layer_name2)

    if X is None or Y is None:
        print(f"Embeddings for layer '{layer_name1}' or '{layer_name2}' are missing.")
        return None, None

    X = X.T
    Y = Y.T

    linear_cka_score = linear_CKA(X, Y)
    rbf_cka_score = kernel_CKA(X, Y)

    return linear_cka_score, rbf_cka_score

def compute_inter_model_cka(exp_dirs, layer_names, metric='linear'):
    """
    Computes inter-model CKA similarities for corresponding layers across multiple models.

    Args:
        exp_dirs (list): List of experiment directory paths (each for one model).
        layer_names (list): List of layer names (assumed common across models).
        metric (str): 'linear' for linear CKA or 'rbf' for kernel CKA.

    Returns:
        dict: Mapping from layer name to a list of CKA similarity scores (one per pair of models).
    """
    models_embeddings = {}
    for exp_dir in exp_dirs:
        models_embeddings[exp_dir] = load_embeddings(exp_dir, layer_names)

    layer_similarities = {layer: [] for layer in layer_names}

    # Iterate over each layer.
    for layer in layer_names:
        # Iterate over all unique pairs of models.
        for (exp1, emb1), (exp2, emb2) in itertools.combinations(models_embeddings.items(), 2):
            # Compute CKA similarity for the given layer from both models.
            linear_cka, rbf_cka = compute_layer_cka(emb1, emb2, layer, layer)
            if metric.lower() == 'linear':
                score = linear_cka
            else:
                score = rbf_cka
            if score is not None:
                layer_similarities[layer].append(score)
    return layer_similarities


def aggregate_inter_model_cka_by_group(experiment_groups, layer_names, metric='rbf'):
    """
    For each group of experiments (given as a dictionary mapping a group label to a list
    of experiment directories), compute the inter-model CKA for each layer and aggregate
    the results (mean and std) across experiments.

    Args:
        experiment_groups (dict): Mapping from group label (e.g., "dreams") to a list of
                                  experiment directory paths.
        layer_names (list): List of layer names (order is preserved).
        metric (str): 'linear' or 'rbf' indicating which CKA metric to use.

    Returns:
        group_results (dict): Mapping from group label to a tuple (means, stds) where each is a
                              1D array (length = number of layers) of the aggregated CKA values.
    """
    group_results = {}
    for group_label, exp_dirs in experiment_groups.items():
        layer_similarities = compute_inter_model_cka(exp_dirs, layer_names, metric=metric)
        means = []
        stds = []
        for layer in layer_names:
            scores = np.array(layer_similarities[layer])
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        group_results[group_label] = (np.array(means), np.array(stds))
    return group_results

def compute_cka_matrix_for_experiment(experiment_dir, layer_names):

    print(f"\nProcessing experiment: {experiment_dir}")
    embeddings = load_embeddings(experiment_dir, layer_names)
    num_layers = len(layer_names)
    cka_matrix = np.zeros((num_layers, num_layers, 2))

    for i, layer1 in enumerate(tqdm(layer_names, desc="Outer loop")):
        for j, layer2 in enumerate(layer_names):
            linear_cka, rbf_cka = compute_layer_cka(embeddings, embeddings, layer1, layer2)
            if linear_cka is not None and rbf_cka is not None:
                cka_matrix[i, j, 0] = linear_cka
                cka_matrix[i, j, 1] = rbf_cka
    return cka_matrix

def compute_all_cka_matrices(experiment_dirs, layer_names):

    cka_matrices = {}
    for exp_dir in experiment_dirs:
        cka_matrices[exp_dir] = compute_cka_matrix_for_experiment(exp_dir, layer_names)
    return cka_matrices


def compute_alignment_metric(X, Y, top_k=5, center=True):
    """
    Computes an aggregate alignment metric between representations X and Y for a given layer.
    It performs shared subspace analysis and returns the average cosine similarity over the top_k eigenvectors.

    Args:
        X (np.ndarray): Activation matrix from model A of shape [num_examples, dim].
        Y (np.ndarray): Activation matrix from model B of shape [num_examples, dim].
        top_k (int): Number of top eigenvectors to consider.
        center (bool): Whether to center the data.

    Returns:
        float: Average cosine similarity over the top_k eigenvectors.
    """
    eigvals, proj_norms, cosines = shared_subspace_analysis(X, Y, center=center)
    top_cos = cosines[:min(top_k, len(cosines))]
    return np.mean(top_cos)


def aggregate_alignment_metric(exp_dirs, layer_names, top_k=5, center=True):
    """
    For a list of experiment directories, compute the alignment metric for each layer
    by comparing each unique pair of experiments using shared subspace analysis.

    Args:
        exp_dirs (list): List of experiment directory paths.
        layer_names (list): List of layer names.
        top_k (int): Number of top eigenvectors to consider.
        center (bool): Whether to center the data.

    Returns:
        agg_dict (dict): Dictionary mapping each layer name to a tuple (mean, std, all_scores),
                         where all_scores is a list of alignment metrics computed across pairs.
    """
    agg_dict = {layer: [] for layer in layer_names}
    models_embeddings = {}
    for exp_dir in exp_dirs:
        models_embeddings[exp_dir] = load_embeddings(exp_dir, layer_names)

    for exp1, exp2 in itertools.combinations(exp_dirs, 2):
        emb1 = models_embeddings[exp1]
        emb2 = models_embeddings[exp2]
        for layer in layer_names:
            if layer in emb1 and layer in emb2:
                X = emb1[layer]  # shape: [num_examples, dim]
                Y = emb2[layer]
                metric = compute_alignment_metric(X, Y, top_k=top_k, center=center)
                agg_dict[layer].append(metric)

    for layer in agg_dict:
        scores = np.array(agg_dict[layer])
        if scores.size > 0:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
        else:
            mean_val, std_val = np.nan, np.nan
        agg_dict[layer] = (mean_val, std_val, agg_dict[layer])
    return agg_dict