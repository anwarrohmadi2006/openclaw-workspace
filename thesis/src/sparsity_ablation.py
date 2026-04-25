"""
Step 9 — Ablasi Sparsity
Test pipeline under different sparsity levels.
"""

import numpy as np
import time
import tracemalloc
from .braid_word import row_to_braid_word
from .theta_eval import theta_eval, compute_theta_features
from .sparse_handler import filter_sparse_features, compute_sparsity


def inject_sparsity(X, target_sparsity, random_state=42):
    """
    Artificially inject sparsity into feature matrix.
    Sets random elements to zero to reach target sparsity.

    Args:
        X: feature matrix
        target_sparsity: fraction of elements to set to 0

    Returns:
        X_sparse with injected zeros
    """
    if target_sparsity == 0:
        return X.copy()

    rng = np.random.RandomState(random_state)
    X_sparse = X.copy()
    mask = rng.random(X_sparse.shape) < target_sparsity
    X_sparse[mask] = 0

    actual = (X_sparse == 0).mean()
    print(f"[SparseAblation] Target: {target_sparsity:.0%}, Actual: {actual:.1%}")
    return X_sparse


def ablation_sparsity(X, feature_order, sparsity_levels=None):
    """
    Run ablation study on sparsity levels.

    Tests:
    - Each sparsity level with and without sparse filtering
    - Measures time, memory, theta distribution

    Returns:
        list of result dicts
    """
    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.2, 0.5, 0.8, 0.95]

    n_samples = min(1000, X.shape[0])  # Limit for speed
    results = []

    for sparsity in sparsity_levels:
        print(f"\n--- Sparsity: {sparsity:.0%} ---")
        X_sparse = inject_sparsity(X, sparsity)

        # Without sparse filtering
        tracemalloc.start()
        start = time.time()
        bws = []
        for i in range(n_samples):
            bw = row_to_braid_word(X_sparse[i], feature_order)
            bws.append(bw)
        time_no_filter = time.time() - start
        _, mem_no_filter = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Theta values
        thetas = [theta_eval(bw, 0.5) for bw in bws]

        results.append({
            "sparsity": sparsity,
            "filter": "none",
            "time_s": time_no_filter,
            "memory_mb": mem_no_filter / 1024 / 1024,
            "theta_mean": np.mean(thetas),
            "theta_std": np.std(thetas),
            "n_features": X_sparse.shape[1],
        })

        # With sparse filtering
        X_filtered, adj_order, kept = filter_sparse_features(X_sparse, threshold=0.9, verbose=False)

        tracemalloc.start()
        start = time.time()
        bws_filtered = []
        for i in range(min(n_samples, X_filtered.shape[0])):
            bw = row_to_braid_word(X_filtered[i], adj_order)
            bws_filtered.append(bw)
        time_filtered = time.time() - start
        _, mem_filtered = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        thetas_filtered = [theta_eval(bw, 0.5) for bw in bws_filtered]

        results.append({
            "sparsity": sparsity,
            "filter": "sparse_filter",
            "time_s": time_filtered,
            "memory_mb": mem_filtered / 1024 / 1024,
            "theta_mean": np.mean(thetas_filtered) if thetas_filtered else 0,
            "theta_std": np.std(thetas_filtered) if thetas_filtered else 0,
            "n_features": X_filtered.shape[1],
        })

        print(f"  No filter: {time_no_filter:.3f}s, {mem_no_filter/1024/1024:.2f}MB, "
              f"θ mean={np.mean(thetas):.2f}")
        print(f"  Filtered:  {time_filtered:.3f}s, {mem_filtered/1024/1024:.2f}MB, "
              f"θ mean={np.mean(thetas_filtered):.2f}" if thetas_filtered else "  Filtered: no features left")

    return results
