"""
Step 2 — Feature Ordering (3 strategies for ablation)
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list


def order_by_mutual_information(X, y, max_features=50):
    """
    Strategi A: Mutual Information ordering.
    Compute MI between each feature and target, sort descending.
    Limit to top max_features for braid word generation.
    """
    mi = mutual_info_classif(X, y, random_state=42)
    order = np.argsort(mi)[::-1]  # descending
    if len(order) > max_features:
        order = order[:max_features]
    print(f"[MI] Top {len(order)} features, MI range: {mi[order[0]]:.4f} - {mi[order[-1]]:.4f}")
    return order, mi


def order_by_correlation_clustering(X, y=None, max_features=50):
    """
    Strategi B: Correlation Clustering ordering.
    Compute correlation matrix → distance → hierarchical clustering → leaf order.
    """
    # Subsample for speed if needed
    if X.shape[1] > 200:
        # Use MI pre-filter to get top features first
        mi = mutual_info_classif(X, y, random_state=42)
        top_idx = np.argsort(mi)[::-1][:200]
        X_sub = X[:, top_idx]
    else:
        top_idx = np.arange(X.shape[1])
        X_sub = X

    corr = np.corrcoef(X_sub, rowvar=False)
    corr = np.clip(corr, -1, 1)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)

    # Ensure symmetry and non-negative
    dist = (dist + dist.T) / 2
    dist = np.maximum(dist, 0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    leaf_order = leaves_list(Z)

    # Map back to original indices
    order = top_idx[leaf_order]
    if len(order) > max_features:
        order = order[:max_features]

    print(f"[CorrCluster] Ordered {len(order)} features via hierarchical clustering")
    return order, None


def order_default(X, y=None, max_features=50):
    """
    Strategi C: Default column order (baseline).
    Just use first max_features columns in original order.
    """
    n = min(max_features, X.shape[1])
    order = np.arange(n)
    print(f"[Default] Using first {n} features in original order")
    return order, None


def get_feature_order(X, y, strategy="MI", max_features=50):
    """
    Get feature ordering by strategy name.
    Returns: (order_indices, mi_scores_or_None)
    """
    strategies = {
        "MI": order_by_mutual_information,
        "CorrCluster": order_by_correlation_clustering,
        "Default": order_default,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    return strategies[strategy](X, y, max_features)
