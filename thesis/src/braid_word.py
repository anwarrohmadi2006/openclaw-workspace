"""
Step 3 — Braid Word Generation
Convert tabular row → permutation → signed braid word via bubble sort.

Ref: Bar-Natan & van der Veen (2025), arXiv:2509.18456
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
        e.g. [1, -2, 3] — only crossings where swap occurs are recorded

    Process:
        1. Extract values in feature_order
        2. Start from identity permutation [0,1,...,d-1]
        3. Bubble sort: ONLY when values[p[j]] > values[p[j+1]]:
           - sign = +1 (positive generator sigma_i)
           - swap p[j] and p[j+1]
           - append sign * (j+1) to braid_word
        4. Non-swaps are NOT recorded — braid word only captures crossings
    """
    values = row_values[feature_order].astype(np.float64)
    n = len(values)
    p = np.arange(n)  # identity permutation

    braid_word = []
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if values[p[j]] > values[p[j + 1]]:
                # Crossing occurs: record positive generator, then swap
                braid_word.append(j + 1)
                p[j], p[j + 1] = p[j + 1], p[j]
            # No else — non-swaps are NOT part of the braid word

    return braid_word


def row_to_signed_braid_word(row_values, feature_order):
    """
    Variant: signed braid word where crossing sign encodes
    relative magnitude difference (positive if large > small).

    This captures richer information than unsigned braid word.
    Use this variant when feature magnitudes are meaningful.
    """
    values = row_values[feature_order].astype(np.float64)
    n = len(values)
    p = np.arange(n)

    braid_word = []
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if values[p[j]] > values[p[j + 1]]:
                # Positive generator: strand j crosses over strand j+1
                braid_word.append(+(j + 1))
                p[j], p[j + 1] = p[j + 1], p[j]
            elif values[p[j]] < values[p[j + 1]]:
                # Negative generator: strand j+1 is dominant
                braid_word.append(-(j + 1))
                # No swap since values are in order

    return braid_word


def generate_braid_words(X, feature_order, max_samples=None, signed=False):
    """
    Generate braid words for all rows in X.

    Args:
        X: feature matrix (N, d)
        feature_order: feature ordering indices
        max_samples: limit number of samples (for large datasets)
        signed: if True, use signed variant (captures magnitude info)

    Returns:
        (braid_words, sample_indices)
    """
    n = X.shape[0]
    if max_samples and n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        idx.sort()
        X_sub = X[idx]
        print(f"[Braid] Subsampled {max_samples} from {n} rows")
    else:
        X_sub = X
        idx = np.arange(n)

    braid_fn = row_to_signed_braid_word if signed else row_to_braid_word

    braid_words = []
    total = X_sub.shape[0]
    for i in range(total):
        bw = braid_fn(X_sub[i], feature_order)
        braid_words.append(bw)

    lengths = [len(b) for b in braid_words]
    print(f"[Braid] Generated {len(braid_words)} braid words | "
          f"avg length: {np.mean(lengths):.1f} | "
          f"max: {np.max(lengths)} | min: {np.min(lengths)}")
    return braid_words, idx
