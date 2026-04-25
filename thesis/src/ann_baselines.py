"""
Priority 4 — ANN Baseline Comparisons
Minimal 2 baseline ANN methods required for Scopus reviewer credibility.

Baseline 1: hnswlib (HNSW — Hierarchical Navigable Small World)
    - State-of-the-art approximate nearest neighbor index.
    - Logarithmic query time, excellent recall/speed trade-off.
    - Reference: Malkov & Yashunin (2018), arXiv:1603.09320.

Baseline 2: Annoy (Approximate Nearest Neighbors Oh Yeah)
    - Tree-based ANN index (random projection forests).
    - Spotify open source. Memory efficient, good for batch queries.
    - Reference: Bernhardsson (2013), github.com/spotify/annoy.

Both baselines are evaluated on:
    (a) Raw normalized features
    (b) Theta-augmented features
using the same Recall@k metric as the main experiments.

Installation (add to requirements.txt):
    hnswlib>=0.7.0
    annoy>=1.17.3
"""

from __future__ import annotations

import numpy as np
import time
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# HNSW baseline (hnswlib)
# ---------------------------------------------------------------------------

def _try_import_hnswlib():
    try:
        import hnswlib
        return hnswlib
    except ImportError:
        return None


def run_hnsw_baseline(
    X_index: np.ndarray,
    y_index: np.ndarray,
    X_query: Optional[np.ndarray] = None,
    y_query: Optional[np.ndarray] = None,
    k_values: Tuple[int, ...] = (1, 5, 10),
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 50,
    space: str = "cosine",
    label: str = "HNSW",
) -> Dict:
    """
    Run HNSW approximate nearest neighbor baseline.

    Args:
        X_index: (N, d) float32 array — used to build the index.
        y_index: (N,) integer labels for the index set.
        X_query: (M, d) query vectors. If None, uses X_index (self-query).
        y_query: (M,) labels for queries. If None, uses y_index.
        k_values: Recall@k values to evaluate.
        ef_construction: HNSW build parameter (higher = better quality, slower).
        M: HNSW connectivity parameter.
        ef_search: HNSW query parameter (higher = better recall, slower).
        space: distance space ('cosine', 'l2', 'ip').
        label: name for this run (used in output dict).

    Returns:
        dict with recall scores, build/query times, and metadata.
    """
    hnswlib = _try_import_hnswlib()
    if hnswlib is None:
        print(f"[HNSW] hnswlib not installed. Falling back to exact brute-force.")
        return _brute_force_fallback(X_index, y_index, X_query, y_query, k_values, label)

    n, dim = X_index.shape
    X_f32 = X_index.astype(np.float32)
    if X_query is None:
        X_query_f32 = X_f32
        y_query = y_index
    else:
        X_query_f32 = X_query.astype(np.float32)

    # Build index
    t_build = time.time()
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(X_f32, np.arange(n))
    index.set_ef(ef_search)
    build_time = time.time() - t_build

    # Query
    max_k = max(k_values)
    t_query = time.time()
    labels_approx, _ = index.knn_query(X_query_f32, k=min(max_k + 1, n))
    query_time = time.time() - t_query

    # Compute recall
    result = {"method": label, "space": space, "build_time_s": build_time,
              "query_time_s": query_time, "n_index": n, "dim": dim}
    for k in k_values:
        recalls = []
        for i in range(len(X_query_f32)):
            nbrs = labels_approx[i]
            # Exclude self if self-query
            if X_query is None:
                nbrs = nbrs[nbrs != i]
            nbrs = nbrs[:k]
            same = np.sum(y_index[nbrs] == y_query[i])
            recalls.append(same / max(len(nbrs), 1))
        result[f"recall@{k}"] = float(np.mean(recalls))
        print(f"  [{label}] recall@{k}: {result[f'recall@{k}']:.4f}")

    print(f"  [{label}] build={build_time:.2f}s, query={query_time:.3f}s")
    return result


# ---------------------------------------------------------------------------
# Annoy baseline
# ---------------------------------------------------------------------------

def _try_import_annoy():
    try:
        from annoy import AnnoyIndex
        return AnnoyIndex
    except ImportError:
        return None


def run_annoy_baseline(
    X_index: np.ndarray,
    y_index: np.ndarray,
    X_query: Optional[np.ndarray] = None,
    y_query: Optional[np.ndarray] = None,
    k_values: Tuple[int, ...] = (1, 5, 10),
    n_trees: int = 50,
    metric: str = "angular",
    label: str = "Annoy",
) -> Dict:
    """
    Run Annoy approximate nearest neighbor baseline.

    Args:
        X_index: (N, d) float array.
        y_index: (N,) integer labels.
        X_query: (M, d) query vectors. If None, uses X_index.
        y_query: (M,) labels. If None, uses y_index.
        k_values: Recall@k values.
        n_trees: number of trees (higher = better recall, more memory).
        metric: 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'.
        label: name for this run.

    Returns:
        dict with recall scores and timing.
    """
    AnnoyIndex = _try_import_annoy()
    if AnnoyIndex is None:
        print(f"[Annoy] annoy not installed. Falling back to exact brute-force.")
        return _brute_force_fallback(X_index, y_index, X_query, y_query, k_values, label)

    n, dim = X_index.shape
    if X_query is None:
        X_query = X_index
        y_query = y_index

    # Build index
    t_build = time.time()
    idx = AnnoyIndex(dim, metric)
    for i in range(n):
        idx.add_item(i, X_index[i].tolist())
    idx.build(n_trees)
    build_time = time.time() - t_build

    # Query
    max_k = max(k_values) + 1
    t_query = time.time()
    all_neighbors = [idx.get_nns_by_vector(X_query[i].tolist(), max_k) for i in range(len(X_query))]
    query_time = time.time() - t_query

    # Compute recall
    result = {"method": label, "metric": metric, "build_time_s": build_time,
              "query_time_s": query_time, "n_index": n, "dim": dim}
    for k in k_values:
        recalls = []
        for i, nbrs in enumerate(all_neighbors):
            nbrs = np.array(nbrs)
            if X_query is X_index:
                nbrs = nbrs[nbrs != i]
            nbrs = nbrs[:k]
            same = np.sum(y_index[nbrs] == y_query[i])
            recalls.append(same / max(len(nbrs), 1))
        result[f"recall@{k}"] = float(np.mean(recalls))
        print(f"  [{label}] recall@{k}: {result[f'recall@{k}']:.4f}")

    print(f"  [{label}] build={build_time:.2f}s, query={query_time:.3f}s")
    return result


# ---------------------------------------------------------------------------
# Brute-force fallback (no extra deps)
# ---------------------------------------------------------------------------

def _brute_force_fallback(
    X_index: np.ndarray,
    y_index: np.ndarray,
    X_query: Optional[np.ndarray],
    y_query: Optional[np.ndarray],
    k_values: Tuple[int, ...],
    label: str,
) -> Dict:
    """Exact brute-force NN used when ANN libraries are unavailable."""
    from sklearn.neighbors import NearestNeighbors
    if X_query is None:
        X_query = X_index
        y_query = y_index

    max_k = max(k_values) + 1
    nn = NearestNeighbors(n_neighbors=min(max_k, len(X_index)), metric="cosine",
                         algorithm="brute", n_jobs=-1)
    t0 = time.time()
    nn.fit(X_index)
    _, indices = nn.kneighbors(X_query)
    elapsed = time.time() - t0

    result = {"method": f"{label}(brute)", "build_time_s": elapsed,
              "query_time_s": elapsed, "n_index": len(X_index)}
    for k in k_values:
        recalls = []
        for i in range(len(X_query)):
            nbrs = indices[i]
            if X_query is X_index:
                nbrs = nbrs[nbrs != i]
            nbrs = nbrs[:k]
            same = np.sum(y_index[nbrs] == y_query[i])
            recalls.append(same / max(len(nbrs), 1))
        result[f"recall@{k}"] = float(np.mean(recalls))
    return result


# ---------------------------------------------------------------------------
# Run both baselines
# ---------------------------------------------------------------------------

def run_ann_baselines(
    X_raw: np.ndarray,
    X_aug: np.ndarray,
    y: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10),
    subsample: int = 5000,
) -> List[Dict]:
    """
    Run both ANN baselines (HNSW + Annoy) on raw and augmented features.

    Args:
        X_raw: original normalized feature matrix (N, d).
        X_aug: theta-augmented feature matrix (N, d+k).
        y: labels (N,).
        k_values: Recall@k values.
        subsample: max samples for ANN evaluation (for speed).

    Returns:
        list of result dicts for all (method, feature_set) combinations.
    """
    # Subsample for speed
    n = len(y)
    if n > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, subsample, replace=False)
        X_raw_, X_aug_, y_ = X_raw[idx], X_aug[idx], y[idx]
        print(f"[ANN] Subsampled {subsample} from {n} for baseline eval")
    else:
        X_raw_, X_aug_, y_ = X_raw, X_aug, y

    results = []
    for X_, name in [(X_raw_, "raw"), (X_aug_, "aug")]:
        print(f"\n[ANN] HNSW on {name} features ({X_.shape})")
        results.append(run_hnsw_baseline(X_, y_, k_values=k_values,
                                         label=f"HNSW_{name}"))
        print(f"\n[ANN] Annoy on {name} features ({X_.shape})")
        results.append(run_annoy_baseline(X_, y_, k_values=k_values,
                                          label=f"Annoy_{name}"))
    return results
