"""
Step 5 — Sparse Data Handling
Filter features with high sparsity before braid word generation.
"""

import numpy as np


def compute_sparsity(X):
    """
    Compute sparsity per feature.
    Sparsity = fraction of zero values.

    Args:
        X: feature matrix (N, d)

    Returns:
        1D array of sparsity per feature
    """
    return (X == 0).mean(axis=0)


def filter_sparse_features(X, threshold=0.9, verbose=True):
    """
    Remove features with sparsity > threshold.

    Args:
        X: feature matrix (N, d)
        threshold: max allowed sparsity (default 0.9)

    Returns:
        X_filtered, kept_indices
    """
    sparsity = compute_sparsity(X)
    kept = np.where(sparsity <= threshold)[0]
    removed = X.shape[1] - len(kept)

    if verbose:
        print(f"[Sparse] Threshold: {threshold}")
        print(f"  Kept: {len(kept)} features, Removed: {removed} features")
        print(f"  Sparsity range kept: [{sparsity[kept].min():.3f}, {sparsity[kept].max():.3f}]")

    return X[:, kept], kept


def apply_sparse_filter(X, feature_order, threshold=0.9):
    """
    Apply sparse filtering and adjust feature order accordingly.

    Args:
        X: feature matrix
        feature_order: original feature ordering
        threshold: sparsity threshold

    Returns:
        X_filtered, adjusted_feature_order, kept_indices
    """
    X_filtered, kept = filter_sparse_features(X, threshold)

    # Map feature_order to filtered indices
    kept_set = set(kept)
    adjusted_order = [idx for idx in feature_order if idx in kept_set]

    # Remap to new indices
    idx_map = {old: new for new, old in enumerate(kept)}
    adjusted_order = [idx_map[idx] for idx in adjusted_order]

    print(f"[Sparse] Adjusted feature order: {len(adjusted_order)} features (was {len(feature_order)})")
    return X_filtered, np.array(adjusted_order), kept
