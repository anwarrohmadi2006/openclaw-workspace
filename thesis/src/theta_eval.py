"""
Step 4 — Theta Evaluation
Approximate Theta invariant from braid word at t=0.5 and t=1/3.
Based on Bar-Natan & van der Veen (2019).
"""

import numpy as np


def theta_eval(braid_word, t):
    """
    Evaluate approximate Theta invariant from a braid word.

    Formula (linear approximation):
        Θ(t) = Σ (t if g > 0 else -(1-t)) for g in braid_word

    Args:
        braid_word: list of signed integers (braid generators)
        t: evaluation point (float)

    Returns:
        float: approximate Θ value
    """
    result = 0.0
    for g in braid_word:
        if g > 0:
            result += t
        else:
            result -= (1 - t)
    return result


def compute_theta_features(braid_words, t_values=(0.5, 1/3)):
    """
    Compute Theta features for all braid words at given t values.

    Args:
        braid_words: list of braid word lists
        t_values: tuple of evaluation points

    Returns:
        numpy array of shape (N, len(t_values))
    """
    n = len(braid_words)
    k = len(t_values)
    theta_matrix = np.zeros((n, k), dtype=np.float64)

    for i, bw in enumerate(braid_words):
        for j, t in enumerate(t_values):
            theta_matrix[i, j] = theta_eval(bw, t)

    print(f"[Theta] Computed {k} theta features for {n} samples")
    print(f"  Θ(t={t_values[0]:.3f}) range: [{theta_matrix[:,0].min():.2f}, {theta_matrix[:,0].max():.2f}]")
    print(f"  Θ(t={t_values[1]:.3f}) range: [{theta_matrix[:,1].min():.2f}, {theta_matrix[:,1].max():.2f}]")
    return theta_matrix
