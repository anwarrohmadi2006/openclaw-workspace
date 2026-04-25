"""
Step 3 — Braid Word Generation
Convert tabular row → permutation → signed braid word via bubble sort.

Definition (canonical):
    A positive generator σ_j is recorded ONLY when the bubble sort makes a
    swap at position j (i.e. values[p[j]] > values[p[j+1]]).
    Non-swap positions are NOT recorded — they contribute no crossing.
    This is the standard braid-word encoding used in topological data analysis.

For the ablation variant that also records negative generators for
non-swaps see `row_to_signed_braid_word()` below.
"""

import numpy as np


def row_to_braid_word(row_values, feature_order):
    """
    Convert a single row to a braid word (positive generators only).

    Only swap-positions are recorded as σ_j (positive integer j).
    Non-swap positions are skipped — they produce no crossing in the braid.

    Args:
        row_values: 1D array of feature values for one sample
        feature_order: indices specifying feature ordering

    Returns:
        list of positive integers (braid generators), e.g. [1, 2, 1, 3]
    """
    values = row_values[feature_order].astype(np.float64)
    n = len(values)
    p = np.arange(n)  # identity permutation

    braid_word = []
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if values[p[j]] > values[p[j + 1]]:
                # Out-of-order pair: record positive generator, then swap
                braid_word.append(j + 1)
                p[j], p[j + 1] = p[j + 1], p[j]
            # In-order: no crossing, nothing recorded

    return braid_word


def row_to_signed_braid_word(row_values, feature_order):
    """
    Ablation variant: records BOTH positive and negative generators.

    Every pair (j, j+1) at every pass produces a generator:
        σ_j   (positive, j+1) if values[p[j]] > values[p[j+1]]  (swap)
        σ_j⁻¹ (negative, -(j+1)) if values[p[j]] <= values[p[j+1]] (no swap)

    Use this for ablation studies to compare against the canonical version.

    Args:
        row_values: 1D array of feature values
        feature_order: feature ordering indices

    Returns:
        list of signed integers, e.g. [1, -2, 1, 3, -1]
    """
    values = row_values[feature_order].astype(np.float64)
    n = len(values)
    p = np.arange(n)

    braid_word = []
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if values[p[j]] > values[p[j + 1]]:
                braid_word.append(j + 1)
                p[j], p[j + 1] = p[j + 1], p[j]
            else:
                braid_word.append(-(j + 1))

    return braid_word


def generate_braid_words(X, feature_order, max_samples=None, signed=False):
    """
    Generate braid words for all rows in X.

    Args:
        X: feature matrix (N, d)
        feature_order: feature ordering indices
        max_samples: optional int — limit rows (random subsample, seed=42)
        signed: if True, use `row_to_signed_braid_word` (ablation variant)

    Returns:
        braid_words: list of lists
        idx: array of row indices used (length = len(braid_words))
    """
    encoder = row_to_signed_braid_word if signed else row_to_braid_word

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

    braid_words = [encoder(X_sub[i], feature_order) for i in range(X_sub.shape[0])]

    lengths = [len(b) for b in braid_words]
    print(f"[Braid] Generated {len(braid_words)} braid words, "
          f"avg length: {np.mean(lengths):.1f}, "
          f"mode: {'signed' if signed else 'canonical'}")
    return braid_words, idx
