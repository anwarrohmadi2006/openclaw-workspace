"""
Similarity Search Benchmark — 3 Kondisi vs 3 Baseline
Based on: Bar-Natan & van der Veen (2025), arXiv:2509.18456

Eksperimen utama:
    Kondisi A: Recall@k dengan raw features saja (baseline ML)
    Kondisi B: Recall@k dengan Theta features saja (pure topological)
    Kondisi C: Recall@k dengan augmented features (raw + Theta) — METODE KAMI

Baseline ANN:
    1. KD-Tree (scikit-learn) — exact, O(n log n)
    2. HNSW via hnswlib      — approximate, O(log n) query
    3. PCA (50 komponen) + cosine similarity — dimensionality reduction baseline

Metrik:
    - Recall@k: fraksi true same-class neighbors dalam top-k
    - Query time per item
    - Index build time
    - Memory usage (tracemalloc)
"""

import numpy as np
import time
import tracemalloc
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Recall@k Helper
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k_from_indices(query_labels: np.ndarray,
                              neighbor_indices: np.ndarray,
                              all_labels: np.ndarray,
                              k: int) -> float:
    """
    Hitung Recall@k dari hasil nearest neighbor search.

    Args:
        query_labels: label untuk setiap query sample (N,)
        neighbor_indices: (N, k) array index tetangga
        all_labels: label seluruh dataset (N,)
        k: jumlah tetangga

    Returns:
        mean Recall@k
    """
    recalls = []
    for i in range(len(query_labels)):
        nbr_labels = all_labels[neighbor_indices[i]]
        same = np.sum(nbr_labels == query_labels[i])
        recalls.append(same / k)
    return float(np.mean(recalls))


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: KD-Tree (scikit-learn)
# ─────────────────────────────────────────────────────────────────────────────

def run_kdtree_baseline(X: np.ndarray, y: np.ndarray, k: int = 10) -> Dict:
    """
    KD-Tree exact nearest neighbor search.
    Kompleksitas: build O(n log n), query O(sqrt(n)) average.
    """
    tracemalloc.start()
    t_build = time.time()
    tree = KDTree(X, leaf_size=40)
    build_time = time.time() - t_build

    t_query = time.time()
    # k+1 karena termasuk diri sendiri
    dist, ind = tree.query(X, k=k + 1)
    query_time = time.time() - t_query
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Skip self (posisi 0)
    neighbor_idx = ind[:, 1:k + 1]
    recall = recall_at_k_from_indices(y, neighbor_idx, y, k)

    return {
        "method": "KDTree",
        "recall_at_k": recall,
        "build_time_s": build_time,
        "query_time_s": query_time,
        "memory_mb": peak_mem / 1024 / 1024,
        "k": k,
        "n_features": X.shape[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: HNSW via hnswlib
# ─────────────────────────────────────────────────────────────────────────────

def run_hnsw_baseline(X: np.ndarray, y: np.ndarray, k: int = 10,
                      ef_construction: int = 200, M: int = 16) -> Dict:
    """
    HNSW approximate nearest neighbor via hnswlib.
    Kompleksitas: build O(n log n), query O(log n).

    Jika hnswlib tidak tersedia, fallback ke brute-force cosine.
    """
    try:
        import hnswlib
        dim = X.shape[1]
        X_norm = normalize(X.astype(np.float32))  # hnswlib butuh float32

        tracemalloc.start()
        t_build = time.time()
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=len(X), ef_construction=ef_construction, M=M)
        index.add_items(X_norm, np.arange(len(X)))
        index.set_ef(50)
        build_time = time.time() - t_build

        t_query = time.time()
        labels, distances = index.knn_query(X_norm, k=k + 1)
        query_time = time.time() - t_query
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Skip self (biasanya di posisi 0, tapi tidak dijamin di HNSW)
        neighbor_idx = np.array([
            [idx for idx in labels[i] if idx != i][:k]
            for i in range(len(X))
        ])
        # Pad jika kurang dari k (edge case)
        for i in range(len(neighbor_idx)):
            if len(neighbor_idx[i]) < k:
                neighbor_idx[i] = np.pad(
                    neighbor_idx[i], (0, k - len(neighbor_idx[i])),
                    constant_values=0
                )
        neighbor_idx = np.array(neighbor_idx)
        recall = recall_at_k_from_indices(y, neighbor_idx, y, k)
        method_name = "HNSW"

    except ImportError:
        # Fallback: brute-force cosine
        print("[HNSW] hnswlib tidak tersedia, fallback ke brute-force cosine")
        from sklearn.metrics.pairwise import cosine_similarity
        tracemalloc.start()
        t_build = time.time()
        sim_matrix = cosine_similarity(X)
        build_time = time.time() - t_build
        np.fill_diagonal(sim_matrix, -1)  # exclude self
        t_query = time.time()
        neighbor_idx = np.argsort(-sim_matrix, axis=1)[:, :k]
        query_time = time.time() - t_query
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        recall = recall_at_k_from_indices(y, neighbor_idx, y, k)
        method_name = "BruteForce-Cosine (HNSW fallback)"

    return {
        "method": method_name,
        "recall_at_k": recall,
        "build_time_s": build_time,
        "query_time_s": query_time,
        "memory_mb": peak_mem / 1024 / 1024,
        "k": k,
        "n_features": X.shape[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: PCA + Cosine Similarity
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_cosine_baseline(X: np.ndarray, y: np.ndarray, k: int = 10,
                             n_components: int = 50) -> Dict:
    """
    PCA dimensionality reduction + cosine similarity.
    Standard dimensionality-reduction baseline untuk similarity search.
    """
    n_components = min(n_components, X.shape[1], X.shape[0])

    tracemalloc.start()
    t_build = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    X_pca_norm = normalize(X_pca)
    build_time = time.time() - t_build

    from sklearn.metrics.pairwise import cosine_similarity
    t_query = time.time()
    sim_matrix = cosine_similarity(X_pca_norm)
    np.fill_diagonal(sim_matrix, -1)
    neighbor_idx = np.argsort(-sim_matrix, axis=1)[:, :k]
    query_time = time.time() - t_query
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recall = recall_at_k_from_indices(y, neighbor_idx, y, k)
    explained_var = pca.explained_variance_ratio_.sum()

    return {
        "method": f"PCA({n_components})+Cosine",
        "recall_at_k": recall,
        "build_time_s": build_time,
        "query_time_s": query_time,
        "memory_mb": peak_mem / 1024 / 1024,
        "k": k,
        "n_features": n_components,
        "explained_variance": float(explained_var),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3 Kondisi Utama (Kontribusi Paper)
# ─────────────────────────────────────────────────────────────────────────────

def run_lgbm_leaf_similarity(
    X: np.ndarray, y: np.ndarray,
    lgbm_model,
    condition_name: str,
    k: int = 10
) -> Dict:
    """
    Recall@k menggunakan LGBM leaf-index embedding.
    Ini adalah metode utama paper untuk similarity search.

    Args:
        X: feature matrix
        y: labels
        lgbm_model: trained LightGBM model
        condition_name: nama kondisi (A/B/C)
        k: top-k

    Returns:
        dict hasil
    """
    from sklearn.neighbors import NearestNeighbors

    tracemalloc.start()
    t_start = time.time()

    leaf_embed = lgbm_model.predict(X, pred_leaf=True)
    if leaf_embed.ndim == 1:
        leaf_embed = leaf_embed.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn.fit(leaf_embed)
    _, indices = nn.kneighbors(leaf_embed)
    elapsed = time.time() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Skip self by position
    neighbor_idx = indices[:, 1:k + 1]
    recall = recall_at_k_from_indices(y, neighbor_idx, y, k)

    return {
        "method": f"LGBM-Leaf [{condition_name}]",
        "recall_at_k": recall,
        "build_time_s": elapsed,
        "query_time_s": elapsed,
        "memory_mb": peak_mem / 1024 / 1024,
        "k": k,
        "n_features": X.shape[1],
        "condition": condition_name,
    }


def run_full_similarity_benchmark(
    X_raw: np.ndarray,
    X_theta_only: np.ndarray,
    X_augmented: np.ndarray,
    y: np.ndarray,
    lgbm_raw,
    lgbm_theta,
    lgbm_augmented,
    k: int = 10,
    max_samples: int = 5000
) -> List[Dict]:
    """
    Jalankan full similarity search benchmark:
    - 3 kondisi LGBM-leaf (A: raw, B: theta-only, C: augmented)
    - 3 baseline (KDTree, HNSW, PCA+Cosine)

    Args:
        X_raw: raw normalized features
        X_theta_only: theta features saja (tanpa raw)
        X_augmented: raw + theta features (augmented)
        y: labels
        lgbm_raw/theta/augmented: trained LGBM models untuk tiap kondisi
        k: top-k
        max_samples: batasi jumlah sample untuk kecepatan

    Returns:
        list of result dicts
    """
    # Subsample untuk efisiensi benchmark
    n = min(max_samples, len(X_raw))
    if n < len(X_raw):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_raw), n, replace=False)
        X_raw_s = X_raw[idx]
        X_theta_s = X_theta_only[idx]
        X_aug_s = X_augmented[idx]
        y_s = y[idx]
        print(f"[SimilarityBench] Subsampled {n}/{len(X_raw)} for benchmark")
    else:
        X_raw_s, X_theta_s, X_aug_s, y_s = X_raw, X_theta_only, X_augmented, y

    results = []

    print("\n── Kondisi A: LGBM + Raw Features ──")
    results.append(run_lgbm_leaf_similarity(X_raw_s, y_s, lgbm_raw, "A-Raw", k))

    print("── Kondisi B: LGBM + Theta Only ──")
    results.append(run_lgbm_leaf_similarity(X_theta_s, y_s, lgbm_theta, "B-ThetaOnly", k))

    print("── Kondisi C: LGBM + Augmented (Raw + Theta) ──")
    results.append(run_lgbm_leaf_similarity(X_aug_s, y_s, lgbm_augmented, "C-Augmented", k))

    print("── Baseline 1: KD-Tree ──")
    results.append(run_kdtree_baseline(X_raw_s, y_s, k))

    print("── Baseline 2: HNSW ──")
    results.append(run_hnsw_baseline(X_raw_s, y_s, k))

    print("── Baseline 3: PCA + Cosine ──")
    results.append(run_pca_cosine_baseline(X_raw_s, y_s, k))

    # Print summary
    print("\n" + "─" * 60)
    print(f"{'Method':<35} {'Recall@' + str(k):<12} {'Time(s)':<10}")
    print("─" * 60)
    for r in results:
        print(f"  {r['method']:<33} {r['recall_at_k']:.4f}       {r['build_time_s']:.3f}s")
    print("─" * 60)

    return results
