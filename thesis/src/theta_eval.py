"""
Step 4 — Theta Evaluation
Approximate Theta invariant from braid word.
Based on Bar-Natan & van der Veen (2025) — arXiv:2509.18456.

Formula (normalized linear approximation):
    n_pos  = number of positive generators (σ_j) in the braid word
    n_neg  = number of negative generators (σ_j⁻¹) in the braid word
    n_total = n_pos + n_neg  (total crossings, 0 → return 0)

    Θ(t) = (n_pos * t  −  n_neg * (1 − t)) / n_total

Range: [-1, 1].  Θ(0.5) = (n_pos − n_neg) / (2 * n_total) ∈ [-0.5, 0.5].
Writhe = n_pos − n_neg  (un-normalized, sign encodes chirality).

Output columns (in order):
    0  Theta_0.5   — balanced evaluation point
    1  Theta_0.333 — skewed toward positive crossings
    2  Theta_0.667 — skewed toward negative crossings
    3  writhe      — signed crossing count (topological invariant proxy)
"""

import numpy as np


def theta_eval(braid_word, t):
    """
    Evaluate normalized approximate Theta invariant.

    Args:
        braid_word: list of signed integers (braid generators).
                    Positive entries = σ_j, negative = σ_j⁻¹.
        t: evaluation point in (0, 1)

    Returns:
        float: Θ(t) ∈ [-1, 1], or 0.0 for empty braid word
    """
    n_pos = sum(1 for g in braid_word if g > 0)
    n_neg = sum(1 for g in braid_word if g < 0)
    n_total = n_pos + n_neg
    if n_total == 0:
        return 0.0
    return (n_pos * t - n_neg * (1.0 - t)) / n_total


def theta_eval_writhe(braid_word):
    """
    Compute writhe = n_pos - n_neg.
    Un-normalized signed crossing count; proxy for chirality of the braid closure.

    Args:
        braid_word: list of signed integers

    Returns:
        int: writhe value
    """
    return sum(1 if g > 0 else -1 for g in braid_word if g != 0)


def compute_theta_features(
    braid_words,
    t_values=(0.5, 1.0 / 3.0, 2.0 / 3.0),
    include_writhe=True,
):
    """
    Compute Theta feature matrix for all braid words.

    Args:
        braid_words: list of braid word lists (length N)
        t_values: tuple of t evaluation points (default: 0.5, 1/3, 2/3)
        include_writhe: if True, append writhe as final column

    Returns:
        numpy array of shape (N, len(t_values) + int(include_writhe))
        Column order: [Theta(t_values[0]), ..., Theta(t_values[-1]), writhe]
    """
    n = len(braid_words)
    k = len(t_values) + int(include_writhe)
    theta_matrix = np.zeros((n, k), dtype=np.float64)

    for i, bw in enumerate(braid_words):
        for j, t in enumerate(t_values):
            theta_matrix[i, j] = theta_eval(bw, t)
        if include_writhe:
            theta_matrix[i, k - 1] = theta_eval_writhe(bw)

    n_theta = len(t_values)
    col_labels = [f"Theta_{t:.3f}" for t in t_values]
    if include_writhe:
        col_labels.append("writhe")

    print(f"[Theta] Computed {k} features ({', '.join(col_labels)}) for {n} samples")
    for j, label in enumerate(col_labels):
        print(f"  {label}: [{theta_matrix[:, j].min():.4f}, {theta_matrix[:, j].max():.4f}]")

    return theta_matrix
