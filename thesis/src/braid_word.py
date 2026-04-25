"""
Step 3 — Braid Word Generation
Convert tabular row → permutation → signed braid word via bubble sort.
"""

import numpy as np


def row_to_braid_word(row_values, feature_order):
    """
    Convert a single row to a signed braid word.

    Args:
        row_values: 1D array of feature values for one sample
        feature_order: indices specifying feature ordering

    Returns:
        list of signed integers representing braid generators
        e.g. [1, -2, 1, 3, -1]

    Process:
        1. Extract values in feature_order
        2. Argsort to get permutation σ
        3. Bubble sort with signed generators:
           - sign = +1 if val[p[j]] > val[p[j+1]] (swap needed)
           - sign = -1 otherwise
    """
    # Extract and argsort
    values = row_values[feature_order].astype(np.float64)
    perm = np.argsort(values)  # ascending order indices

    # Bubble sort decomposition with signs
    braid_word = []
    p = perm.copy()
    n = len(p)

    for i in range(n - 1, 0, -1):
        for j in range(i):
            # Compare values at positions j and j+1
            val_j = values[p[j]]
            val_j1 = values[p[j + 1]]

            if val_j > val_j1:
                # Swap needed: positive generator
                braid_word.append(j + 1)  # 1-indexed generator
                p[j], p[j + 1] = p[j + 1], p[j]
            else:
                # No swap: negative generator
                braid_word.append(-(j + 1))

    return braid_word


def generate_braid_words(X, feature_order, max_samples=None):
    """
    Generate braid words for all rows in X.

    Args:
        X: feature matrix (N, d)
        feature_order: feature ordering indices
        max_samples: limit number of samples (for large datasets)

    Returns:
        list of braid words (list of lists)
    """
    n = X.shape[0]
    if max_samples and n > max_samples:
        # Stratified subsample would be better, but random for now
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        idx.sort()
        X_sub = X[idx]
        print(f"[Braid] Subsampled {max_samples} from {n} rows")
    else:
        X_sub = X
        idx = np.arange(n)

    braid_words = []
    total = X_sub.shape[0]
    for i in range(total):
        bw = row_to_braid_word(X_sub[i], feature_order)
        braid_words.append(bw)

    print(f"[Braid] Generated {len(braid_words)} braid words, "
          f"avg length: {np.mean([len(b) for b in braid_words]):.1f}")
    return braid_words, idx
