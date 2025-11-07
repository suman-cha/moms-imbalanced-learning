"""
Density Peak Clustering-based Selection for Imbalanced Learning.

This module implements clustering-based selection strategies for handling
imbalanced datasets using Density Peak Clustering (DPC) algorithm.

The selection strategy balances density and distance criteria to select
representative majority class samples for ensemble learning.

References
----------
.. [1] Rodriguez, A., & Laio, A. (2014). "Clustering by fast search and find
       of density peaks." Science, 344(6191), 1492-1496.
"""

from typing import List, Tuple, Optional
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add pydpc to path if not already importable
try:
    from pydpc.dpc import Cluster
except ImportError:
    # Try adding local pydpc directory to path
    project_root = Path(__file__).parent.parent.parent
    pydpc_path = project_root / "pydpc"
    if pydpc_path.exists():
        sys.path.insert(0, str(pydpc_path))
        try:
            from pydpc.dpc import Cluster
        except ImportError:
            raise ImportError(
                "pydpc is required for clustering selection. "
                "Local pydpc found but could not import. "
                "The package may need to be built."
            )
    else:
        raise ImportError(
            "pydpc is required for clustering selection. "
            "Install it with: pip install pydpc"
        )



def _compute_distance_threshold(
    delta_values: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Compute distance threshold using delta values.
    
    Parameters
    ----------
    delta_values : np.ndarray of shape (n_samples,)
        Sorted delta values from density peak clustering.
    threshold : float, default=0.1
        Minimum gap threshold for detecting significant jumps.
        
    Returns
    -------
    distance_threshold : float
        Computed distance threshold.
    """
    n = len(delta_values)
    for d in range(n - 1):
        gap = delta_values[d + 1] - delta_values[d]
        if gap >= threshold:
            return delta_values[d] + 0.01
    return delta_values[-1] + 0.01


def _compute_density_threshold(
    density_values: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    Compute density threshold using density values.
    
    Parameters
    ----------
    density_values : np.ndarray of shape (n_samples,)
        Sorted density values from density peak clustering.
    threshold : float, default=0.01
        Minimum gap threshold for detecting significant jumps.
        
    Returns
    -------
    density_threshold : float
        Computed density threshold.
    """
    n = len(density_values)
    for d in range(n - 1):
        gap = density_values[d + 1] - density_values[d]
        if gap >= threshold:
            return density_values[d] + 0.01
    return density_values[-1] + 0.01


def _compute_cluster_indices(
    membership: np.ndarray,
    n_clusters: int
) -> List[np.ndarray]:
    """
    Compute indices of instances for each cluster.
    
    Parameters
    ----------
    membership : np.ndarray of shape (n_samples,)
        Cluster membership array where each element indicates the cluster ID.
    n_clusters : int
        Total number of clusters.
        
    Returns
    -------
    cluster_indices : List[np.ndarray]
        List where each element contains indices of samples in that cluster.
    """
    cluster_indices = []
    for cluster_id in range(n_clusters):
        indices = np.where(membership == cluster_id)[0]
        cluster_indices.append(indices)
    return cluster_indices


def _compute_cluster_densities(
    density: np.ndarray,
    cluster_indices: List[np.ndarray]
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute density values for instances in each cluster.
    
    Parameters
    ----------
    density : np.ndarray of shape (n_samples,)
        Density value for each sample.
    cluster_indices : List[np.ndarray]
        List of cluster indices.
        
    Returns
    -------
    cluster_instance_densities : List[np.ndarray]
        Density values for each instance in each cluster.
    cluster_total_densities : np.ndarray of shape (n_clusters,)
        Total density for each cluster (sum of instance densities).
    """
    cluster_instance_densities = []
    cluster_total_densities = []
    
    for indices in cluster_indices:
        instance_densities = density[indices]
        cluster_instance_densities.append(instance_densities)
        cluster_total_densities.append(np.sum(instance_densities))
    
    return cluster_instance_densities, np.array(cluster_total_densities)


def _compute_centroid_distances(
    X_majority: np.ndarray,
    X_minority: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Compute distances from cluster centroids to minority class samples.
    
    The distance metric is the sum of L2 distances from each centroid
    to all minority class samples.
    
    Parameters
    ----------
    X_majority : np.ndarray of shape (n_majority_samples, n_features)
        Majority class samples.
    X_minority : np.ndarray of shape (n_minority_samples, n_features)
        Minority class samples.
    centroids : np.ndarray of shape (n_clusters,)
        Indices of cluster centroids in X_majority.
        
    Returns
    -------
    cluster_distances : np.ndarray of shape (n_clusters,)
        Total distance from each cluster centroid to all minority samples,
        sorted in descending order.
    """
    n_minority = X_minority.shape[0]
    cluster_distances = []
    
    for centroid_idx in centroids:
        centroid = X_majority[centroid_idx]
        # Compute total distance to all minority samples
        total_dist = np.sum([
            np.linalg.norm(centroid - X_minority[i])
            for i in range(n_minority)
        ])
        cluster_distances.append(total_dist)
    
    cluster_distances = np.array(cluster_distances)
    cluster_distances.sort()  # Sort in ascending order
    cluster_distances = cluster_distances[::-1]  # Reverse to descending
    
    return cluster_distances


def clustering_dpc(
    X_train_majority: np.ndarray,
    X_train_minority: np.ndarray,
    y_train_majority: np.ndarray,
    y_train_minority: np.ndarray,
    density_fraction: float = 0.001,
    delta_threshold: float = 0.1,
    density_threshold: float = 0.01
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Perform Density Peak Clustering on majority class samples.
    
    This function applies the Density Peak Clustering (DPC) algorithm [1]
    to identify clusters in the majority class. The clustering results
    are used to select representative samples for ensemble learning.
    
    Parameters
    ----------
    X_train_majority : np.ndarray of shape (n_majority_samples, n_features)
        Training samples from the majority class.
    X_train_minority : np.ndarray of shape (n_minority_samples, n_features)
        Training samples from the minority class.
    y_train_majority : np.ndarray of shape (n_majority_samples,)
        Labels for majority class samples.
    y_train_minority : np.ndarray of shape (n_minority_samples,)
        Labels for minority class samples.
    density_fraction : float, default=0.001
        Fraction parameter for DPC algorithm controlling local density estimation.
        Valid range: (0.0, 1.0].
    delta_threshold : float, default=0.1
        Threshold for detecting significant gaps in delta values.
    density_threshold : float, default=0.01
        Threshold for detecting significant gaps in density values.
        
    Returns
    -------
    cluster_indices : List[np.ndarray]
        List of arrays containing indices of samples in each cluster.
    clusters_density_normalized : np.ndarray of shape (n_clusters,)
        Normalized density scores for each cluster (sum to 1.0).
    cluster_distances_normalized : np.ndarray of shape (n_clusters,)
        Normalized distance scores for each cluster (sum to 1.0).
    cluster_instance_densities : List[np.ndarray]
        Density values for each instance in each cluster.
        
    References
    ----------
    .. [1] Rodriguez, A., & Laio, A. (2014). "Clustering by fast search and
           find of density peaks." Science, 344(6191), 1492-1496.
           
    Notes
    -----
    The function uses two DPC instances to compute thresholds, which is
    necessary for the adaptive threshold selection strategy.
    """
    # Validate inputs
    if not 0 < density_fraction <= 1.0:
        raise ValueError(
            f"density_fraction must be in (0, 1], got {density_fraction}"
        )
    
    # Initialize DPC clusterers
    dpc = Cluster(X_train_majority, fraction=density_fraction, autoplot=False)
    dpc_threshold = Cluster(
        X_train_majority, fraction=density_fraction, autoplot=False
    )
    
    # Get delta and density values for threshold computation
    delta_values = dpc_threshold.delta.copy()
    density_values = dpc_threshold.density.copy()
    
    delta_values.sort()
    density_values.sort()
    
    # Compute adaptive thresholds
    distance_threshold = _compute_distance_threshold(
        delta_values, delta_threshold
    )
    density_threshold_value = _compute_density_threshold(
        density_values, density_threshold
    )
    
    # Assign clusters based on computed thresholds
    dpc.assign(density_threshold_value, distance_threshold)
    
    n_clusters = dpc.clusters.shape[0]
    
    # Compute cluster properties
    cluster_indices = _compute_cluster_indices(dpc.membership, n_clusters)
    cluster_instance_densities, cluster_total_densities = (
        _compute_cluster_densities(dpc.density, cluster_indices)
    )
    cluster_distances = _compute_centroid_distances(
        X_train_majority, X_train_minority, dpc.clusters
    )
    
    # Normalize scores
    density_sum = np.sum(cluster_total_densities)
    distance_sum = np.sum(cluster_distances)
    
    # Avoid division by zero
    clusters_density_normalized = (
        cluster_total_densities / density_sum
        if density_sum > 0
        else np.zeros_like(cluster_total_densities)
    )
    cluster_distances_normalized = (
        cluster_distances / distance_sum
        if distance_sum > 0
        else np.zeros_like(cluster_distances)
    )
    
    return (
        cluster_indices,
        clusters_density_normalized,
        cluster_distances_normalized,
        cluster_instance_densities
    )

def _compute_selection_weights(
    cluster_densities: np.ndarray,
    cluster_distances: np.ndarray,
    alpha: float,
    beta: float
) -> np.ndarray:
    """
    Compute selection weights for clusters using density and distance.
    
    The selection weight combines density and distance information:
    
    .. math::
        z_i = \\alpha \\cdot d_i + \\beta \\cdot dist_i
    
    where :math:`d_i` is the normalized density and :math:`dist_i` is
    the normalized distance for cluster :math:`i`.
    
    Parameters
    ----------
    cluster_densities : np.ndarray of shape (n_clusters,)
        Normalized density scores for each cluster.
    cluster_distances : np.ndarray of shape (n_clusters,)
        Normalized distance scores for each cluster.
    alpha : float
        Weight for density component. Valid range: [0.0, 1.0].
    beta : float
        Weight for distance component. Valid range: [0.0, 1.0].
        
    Returns
    -------
    selection_weights : np.ndarray of shape (n_clusters,)
        Combined selection weights for each cluster.
        
    Notes
    -----
    Typically, alpha + beta = 1.0 to maintain proper weighting,
    but this is not strictly enforced.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    
    return alpha * cluster_densities + beta * cluster_distances


def _compute_samples_per_cluster(
    selection_weights: np.ndarray,
    n_minority: int,
    cluster_indices: List[np.ndarray],
    sampling_multiplier: int = 6
) -> np.ndarray:
    """
    Compute number of samples to select from each cluster.
    
    The number of samples is proportional to the selection weight and
    aims to generate a dataset with balanced classes.
    
    Parameters
    ----------
    selection_weights : np.ndarray of shape (n_clusters,)
        Selection weight for each cluster.
    n_minority : int
        Number of minority class samples.
    cluster_indices : List[np.ndarray]
        Indices for each cluster.
    sampling_multiplier : int, default=6
        Multiplier for determining total samples to select.
        Total target = n_minority * sampling_multiplier.
        
    Returns
    -------
    n_samples_per_cluster : np.ndarray of shape (n_clusters,)
        Number of samples to select from each cluster.
    """
    n_clusters = len(selection_weights)
    weight_sum = np.sum(selection_weights)
    
    if weight_sum == 0:
        return np.zeros(n_clusters, dtype=int)
    
    # Target total number of majority samples to select
    total_target = n_minority * sampling_multiplier
    
    # Allocate samples proportional to weights
    n_samples_per_cluster = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        # Proportional allocation
        target = (selection_weights[i] * total_target) / weight_sum
        target_rounded = int(np.round(target))
        
        # Ensure we don't select more than available in cluster
        cluster_size = cluster_indices[i].shape[0]
        n_samples_per_cluster[i] = min(target_rounded, cluster_size)
    
    return n_samples_per_cluster


def _select_samples_from_clusters(
    cluster_indices: List[np.ndarray],
    cluster_instance_densities: List[np.ndarray],
    n_samples_per_cluster: np.ndarray
) -> List[int]:
    """
    Select samples from each cluster based on density ranking.
    
    Within each cluster, samples with highest density are selected.
    
    Parameters
    ----------
    cluster_indices : List[np.ndarray]
        Indices for each cluster.
    cluster_instance_densities : List[np.ndarray]
        Density values for instances in each cluster.
    n_samples_per_cluster : np.ndarray of shape (n_clusters,)
        Number of samples to select from each cluster.
        
    Returns
    -------
    selected_indices : List[int]
        Flattened list of selected sample indices.
    """
    selected_per_cluster = []
    
    for cluster_id, n_select in enumerate(n_samples_per_cluster):
        if n_select == 0:
            continue
            
        # Get density values for this cluster
        densities = cluster_instance_densities[cluster_id]
        
        # Sort by density (descending) and select top n_select
        sorted_idx = np.argsort(densities)[::-1][:n_select]
        selected_per_cluster.append(sorted_idx)
    
    # Flatten the list
    selected_indices = [
        idx for cluster_selection in selected_per_cluster
        for idx in cluster_selection
    ]
    
    return selected_indices


def selection(
    X_train_majority: np.ndarray,
    X_train_minority: np.ndarray,
    y_train_majority: np.ndarray,
    y_train_minority: np.ndarray,
    cluster_indices: List[np.ndarray],
    clusters_density: np.ndarray,
    cluster_distances: np.ndarray,
    alpha: float,
    beta: float,
    cluster_instance_densities: List[np.ndarray],
    sampling_multiplier: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select representative samples from majority class using clustering.
    
    This function implements a density and distance-based selection strategy
    to create a balanced training set. The selection process:
    
    1. Computes combined weights using density and distance: z_i = α·d_i + β·dist_i
    2. Allocates samples to clusters proportional to their weights
    3. Selects highest-density samples within each cluster
    4. Combines selected majority samples with all minority samples
    
    Parameters
    ----------
    X_train_majority : np.ndarray of shape (n_majority_samples, n_features)
        Majority class training samples.
    X_train_minority : np.ndarray of shape (n_minority_samples, n_features)
        Minority class training samples.
    y_train_majority : np.ndarray of shape (n_majority_samples,)
        Labels for majority class samples.
    y_train_minority : np.ndarray of shape (n_minority_samples,)
        Labels for minority class samples.
    cluster_indices : List[np.ndarray]
        Indices of samples in each cluster.
    clusters_density : np.ndarray of shape (n_clusters,)
        Normalized density score for each cluster.
    cluster_distances : np.ndarray of shape (n_clusters,)
        Normalized distance score for each cluster.
    alpha : float
        Weight for density component in [0, 1].
    beta : float
        Weight for distance component in [0, 1].
    cluster_instance_densities : List[np.ndarray]
        Density values for instances in each cluster.
    sampling_multiplier : int, default=6
        Multiplier for target number of majority samples.
        Target = n_minority * sampling_multiplier.
        
    Returns
    -------
    X_balanced : np.ndarray of shape (n_balanced_samples, n_features)
        Balanced training set with selected majority and all minority samples.
    y_balanced : np.ndarray of shape (n_balanced_samples,)
        Labels for balanced training set.
    selected_indices : np.ndarray of shape (n_selected_majority,)
        Indices of selected majority class samples.
        
    Examples
    --------
    >>> # After clustering with clustering_dpc
    >>> X_bal, y_bal, indices = selection(
    ...     X_majority, X_minority, y_majority, y_minority,
    ...     cluster_idx, density_scores, distance_scores,
    ...     alpha=0.5, beta=0.5, instance_densities
    ... )
    """
    # Validate inputs
    n_clusters = len(clusters_density)
    
    if len(cluster_distances) != n_clusters:
        raise ValueError(
            f"Mismatch: {n_clusters} clusters but "
            f"{len(cluster_distances)} distance scores"
        )
    
    if not np.isclose(alpha + beta, 1.0):
        import warnings
        warnings.warn(
            f"alpha + beta = {alpha + beta:.4f}, typically should sum to 1.0",
            UserWarning
        )
    
    # Compute selection weights
    selection_weights = _compute_selection_weights(
        clusters_density, cluster_distances, alpha, beta
    )
    
    # Determine number of samples per cluster
    n_minority = X_train_minority.shape[0]
    n_samples_per_cluster = _compute_samples_per_cluster(
        selection_weights, n_minority, cluster_indices, sampling_multiplier
    )
    
    # Select samples from clusters
    selected_flat_indices = _select_samples_from_clusters(
        cluster_indices, cluster_instance_densities, n_samples_per_cluster
    )
    
    # Convert to array
    selected_indices = np.array(selected_flat_indices, dtype=int)
    
    # Extract selected samples
    if len(selected_indices) > 0:
        X_selected = X_train_majority[selected_indices]
        y_selected = y_train_majority[selected_indices]
    else:
        # Handle edge case of no samples selected
        X_selected = np.empty((0, X_train_majority.shape[1]))
        y_selected = np.empty(0, dtype=y_train_majority.dtype)
    
    # Ensure minority labels are array
    y_train_minority = np.asarray(y_train_minority)
    
    # Combine majority and minority samples
    X_balanced = np.concatenate((X_selected, X_train_minority), axis=0)
    y_balanced = np.concatenate((y_selected, y_train_minority), axis=0)
    
    return X_balanced, y_balanced, selected_indices