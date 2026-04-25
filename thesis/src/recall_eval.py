"""
Priority 3 — Recall@k Evaluation (3 Separate Conditions)

Three evaluation conditions for retrieval quality, as required for
the paper's main experiment table:

  Condition A — LGBM leaf embedding similarity
      Use LightGBM leaf indices as embeddings (existing approach).
      Recall@k = fraction of k nearest neighbors sharing the same class.

  Condition B — Raw feature cosine similarity (no embedding)
      Use the raw (or theta-augmented) feature vector directly.
      Baseline: how well does cosine similarity on features retrieve
      same-class neighbors?

  Condition C — Theta-only similarity
      Use only the Theta/Alexander features as the embedding.
      Isolates the contribution of the topological features alone.

Each condition is evaluated with cosine and L2 distance metrics.
Recall@k is computed for k ∈ {1, 5, 10} by default.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Core recall computation
# ---------------------------------------------------------------------------

def recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
    exclude_self: bool = True,
) -> float:
    """
    Compute mean Recall@k over all queries.

    Recall@k for query i = (# same-class neighbors in top-k) / k.

    Args:
        embeddings: (N, d) feature matrix used for nearest-neighbor search.
        labels:     (N,) integer class labels.
        k:          number of neighbors to retrieve.
        metric:     distance metric ('cosine', 'euclidean', 'l2').
        exclude_self: if True, exclude the query point from its own results.

    Returns:
        float: mean Recall@k across all N queries.
    """
    n = len(labels)
    n_neighbors = k + 1 if exclude_self else k
    n_neighbors = min(n_neighbors, n)  # guard against tiny datasets

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    recalls = []
    for i in range(n):
        nbrs = indices[i]
        if exclude_self:
            nbrs = nbrs[nbrs != i][:k]
        same = np.sum(labels[nbrs] == labels[i])
        recalls.append(same / max(len(nbrs), 1))

    return float(np.mean(recalls))


def recall_at_k_multi(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
    metric: str = "cosine",
) -> Dict[str, float]:
    """
    Compute Recall@k for multiple k values in one pass.

    Args:
        embeddings: (N, d) array.
        labels: (N,) integer array.
        k_values: tuple of k values to evaluate.
        metric: distance metric.

    Returns:
        dict mapping 'recall@k' -> float for each k in k_values.
    """
    max_k = max(k_values)
    n = len(labels)
    n_neighbors = min(max_k + 1, n)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    results = {}
    for k in k_values:
        recalls = []
        for i in range(n):
            nbrs = indices[i]
            nbrs = nbrs[nbrs != i][:k]
            same = np.sum(labels[nbrs] == labels[i])
            recalls.append(same / max(len(nbrs), 1))
        results[f"recall@{k}"] = float(np.mean(recalls))

    return results


# ---------------------------------------------------------------------------
# Three evaluation conditions
# ---------------------------------------------------------------------------

def evaluate_recall_condition_a(
    model,
    X: np.ndarray,
    y: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
    metric: str = "cosine",
) -> Dict[str, object]:
    """
    Condition A: LGBM leaf embedding similarity.

    Extracts leaf indices from a trained LightGBM model and uses them as
    embedding vectors for nearest-neighbor retrieval.

    Args:
        model: trained LightGBM booster.
        X: feature matrix (N, d).
        y: labels (N,).
        k_values: Recall@k values to compute.
        metric: distance metric.

    Returns:
        dict with recall scores and condition metadata.
    """
    leaf_embed = model.predict(X, pred_leaf=True)
    if leaf_embed.ndim == 1:
        leaf_embed = leaf_embed.reshape(-1, 1)
    leaf_embed = leaf_embed.astype(np.float64)

    scores = recall_at_k_multi(leaf_embed, y, k_values=k_values, metric=metric)
    print(f"[RecallA] LGBM leaf — " + ", ".join(f"{k}: {v:.4f}" for k, v in scores.items()))
    return {"condition": "A_lgbm_leaf", "metric": metric, **scores}


def evaluate_recall_condition_b(
    X: np.ndarray,
    y: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
    metric: str = "cosine",
) -> Dict[str, object]:
    """
    Condition B: Raw/augmented feature cosine similarity.

    Uses the feature matrix directly (no model embedding).
    If X already contains Theta-augmented features, this evaluates
    the full augmented space.

    Args:
        X: feature matrix (N, d) — raw or theta-augmented.
        y: labels (N,).
        k_values: Recall@k values.
        metric: distance metric.

    Returns:
        dict with recall scores and condition metadata.
    """
    scores = recall_at_k_multi(X, y, k_values=k_values, metric=metric)
    print(f"[RecallB] Raw features — " + ", ".join(f"{k}: {v:.4f}" for k, v in scores.items()))
    return {"condition": "B_raw_features", "metric": metric, **scores}


def evaluate_recall_condition_c(
    theta_features: np.ndarray,
    y: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
    metric: str = "cosine",
) -> Dict[str, object]:
    """
    Condition C: Theta/Alexander features only.

    Evaluates how well the topological features alone retrieve same-class
    neighbors, without any gradient boosting or original features.
    Isolates the contribution of the Theta invariant features.

    Args:
        theta_features: (N, k) matrix of Theta/Alexander features only.
        y: labels (N,).
        k_values: Recall@k values.
        metric: distance metric.

    Returns:
        dict with recall scores and condition metadata.
    """
    scores = recall_at_k_multi(theta_features, y, k_values=k_values, metric=metric)
    print(f"[RecallC] Theta-only — " + ", ".join(f"{k}: {v:.4f}" for k, v in scores.items()))
    return {"condition": "C_theta_only", "metric": metric, **scores}


def evaluate_all_recall_conditions(
    model,
    X: np.ndarray,
    X_aug: np.ndarray,
    theta_features: np.ndarray,
    y: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
    metrics: Tuple[str, ...] = ("cosine",),
) -> List[Dict]:
    """
    Run all three recall conditions and return results as a list.

    Args:
        model: trained LightGBM booster.
        X: original normalized feature matrix (N, d).
        X_aug: theta-augmented feature matrix (N, d+k).
        theta_features: theta-only matrix (N, k).
        y: labels (N,).
        k_values: Recall@k values to evaluate.
        metrics: distance metrics to use.

    Returns:
        list of result dicts, one per (condition, metric) combination.
    """
    results = []
    for metric in metrics:
        results.append(evaluate_recall_condition_a(model, X_aug, y, k_values, metric))
        results.append(evaluate_recall_condition_b(X_aug, y, k_values, metric))
        results.append(evaluate_recall_condition_c(theta_features, y, k_values, metric))
    return results
