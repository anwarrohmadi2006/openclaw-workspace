"""
Step 8 — Computational Efficiency Benchmark
Measure scaling of argsort, braid generation, and theta evaluation.
"""

import numpy as np
import time
import tracemalloc
from .braid_word import row_to_braid_word
from .theta_eval import theta_eval


def benchmark_argsort(X, n_values):
    """Benchmark np.argsort for different N."""
    results = []
    for n in n_values:
        if n > X.shape[0]:
            n = X.shape[0]
        X_sub = X[:n]

        tracemalloc.start()
        start = time.time()
        for i in range(n):
            _ = np.argsort(X_sub[i])
        elapsed = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({"N": n, "operation": "argsort", "time_s": elapsed, "memory_mb": peak_mem / 1024 / 1024})
        print(f"  [Argsort] N={n}: {elapsed:.3f}s, {peak_mem/1024/1024:.2f}MB")
    return results


def benchmark_braid_generation(X, feature_order, n_values):
    """Benchmark braid word generation for different N."""
    results = []
    for n in n_values:
        if n > X.shape[0]:
            n = X.shape[0]
        X_sub = X[:n]

        tracemalloc.start()
        start = time.time()
        for i in range(n):
            _ = row_to_braid_word(X_sub[i], feature_order)
        elapsed = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({"N": n, "operation": "braid_gen", "time_s": elapsed, "memory_mb": peak_mem / 1024 / 1024})
        print(f"  [Braid] N={n}: {elapsed:.3f}s, {peak_mem/1024/1024:.2f}MB")
    return results


def benchmark_theta_eval(braid_words, n_values, t=0.5):
    """Benchmark theta evaluation for different N."""
    results = []
    for n in n_values:
        if n > len(braid_words):
            n = len(braid_words)

        tracemalloc.start()
        start = time.time()
        for i in range(n):
            _ = theta_eval(braid_words[i], t)
        elapsed = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({"N": n, "operation": "theta_eval", "time_s": elapsed, "memory_mb": peak_mem / 1024 / 1024})
        print(f"  [Theta] N={n}: {elapsed:.3f}s, {peak_mem/1024/1024:.2f}MB")
    return results


def run_efficiency_benchmark(X, feature_order, braid_words=None):
    """
    Run full efficiency benchmark.
    Returns list of result dicts.
    """
    n_values = [1000, 5000, 10000, 50000]
    n_values = [n for n in n_values if n <= X.shape[0]]
    if not n_values:
        n_values = [X.shape[0]]

    print(f"\n[Efficiency] Benchmarking for N={n_values}")

    results = []
    results.extend(benchmark_argsort(X, n_values))
    results.extend(benchmark_braid_generation(X, feature_order, n_values))

    if braid_words is None:
        # Generate braid words for max N
        max_n = max(n_values)
        braid_words = []
        for i in range(min(max_n, X.shape[0])):
            braid_words.append(row_to_braid_word(X[i], feature_order))

    results.extend(benchmark_theta_eval(braid_words, n_values))
    return results
