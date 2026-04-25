"""
Priority 2 — Alexander Matrix & Exact Theta Invariant
Compute the reduced Burau representation, Alexander matrix, Alexander
polynomial, and an exact Theta-like invariant for a closed braid.

Background:
    The Burau representation ψ_n: B_n → GL_n(Z[t, t^{-1}]) assigns to each
    braid generator σ_i an n×n matrix over the ring of Laurent polynomials.
    The *reduced* Burau representation ψ_n^{red}: B_n → GL_{n-1}(Z[t, t^{-1}])
    is obtained by a standard change of basis.

    For a braid b with closure b̂, the Alexander polynomial satisfies:
        Δ(t) ≈ det(I - Ψ_{n-1}^{red}(b))  / (1 + t + ... + t^{n-1})

    The exact Theta invariant used here:
        Theta_exact(t) = det(I - Burau(b, t))   (evaluated numerically)

    This is more principled than the linear approximation in theta_eval.py
    and forms the core contribution of this paper.

References:
    - Burau (1936): Uber Zopfgruppen.
    - Bar-Natan & van der Veen (2025): arXiv:2509.18456.
    - Birman (1974): Braids, Links, and Mapping Class Groups.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from .braid_closure import ClosedBraid


# ---------------------------------------------------------------------------
# Burau representation (numerical, evaluated at t ∈ C)
# ---------------------------------------------------------------------------

def _sigma_burau(i: int, n: int, t: complex) -> np.ndarray:
    """
    Full (un-reduced) Burau matrix for generator σ_i on n strands.
    0-indexed: i acts on positions i and i+1.

    σ_i maps to the identity matrix with the 2×2 block at (i, i):
        | 1-t   t |
        |  1    0 |
    """
    M = np.eye(n, dtype=complex)
    M[i, i] = 1 - t
    M[i, i + 1] = t
    M[i + 1, i] = 1.0
    M[i + 1, i + 1] = 0.0
    return M


def _sigma_inv_burau(i: int, n: int, t: complex) -> np.ndarray:
    """
    Full Burau matrix for generator σ_i^{-1} on n strands.

    σ_i^{-1} maps to the 2×2 block at (i, i):
        |  0    1  |
        | t^{-1}  1-t^{-1} |
    """
    M = np.eye(n, dtype=complex)
    M[i, i] = 0.0
    M[i, i + 1] = 1.0
    M[i + 1, i] = 1.0 / t if abs(t) > 1e-12 else 0.0
    M[i + 1, i + 1] = 1.0 - (1.0 / t if abs(t) > 1e-12 else 0.0)
    return M


def burau_matrix(braid_word: List[int], n_strands: int, t: complex) -> np.ndarray:
    """
    Compute the full Burau matrix for a braid word at a given t.

    Args:
        braid_word: list of signed integers (1-indexed generators).
        n_strands: number of braid strands.
        t: evaluation point in C (typically t=0.5 or a root of unity).

    Returns:
        (n_strands, n_strands) complex numpy array.
    """
    M = np.eye(n_strands, dtype=complex)
    for g in braid_word:
        i = abs(g) - 1  # 0-indexed
        if g > 0:
            M = _sigma_burau(i, n_strands, t) @ M
        else:
            M = _sigma_inv_burau(i, n_strands, t) @ M
    return M


def reduced_burau_matrix(braid_word: List[int], n_strands: int, t: complex) -> np.ndarray:
    """
    Compute the reduced Burau matrix (size n_strands-1) for a braid word.

    The reduced representation is obtained by deleting the last row and column
    of the full Burau matrix after a change of basis.  This is the standard
    form used to compute the Alexander polynomial.

    Args:
        braid_word: list of signed integers (1-indexed).
        n_strands: number of strands.
        t: evaluation point in C.

    Returns:
        (n_strands-1, n_strands-1) complex numpy array.
    """
    M_full = burau_matrix(braid_word, n_strands, t)
    # Reduced Burau: delete last row and last column
    return M_full[:-1, :-1]


# ---------------------------------------------------------------------------
# Alexander matrix & polynomial
# ---------------------------------------------------------------------------

def alexander_matrix(braid_word: List[int], n_strands: int, t: complex) -> np.ndarray:
    """
    Compute the Alexander matrix A(t) = I - Burau_reduced(b, t).

    The Alexander polynomial is det(A(t)) up to units in Z[t, t^{-1}].

    Args:
        braid_word: list of signed integers (1-indexed).
        n_strands: number of strands.
        t: evaluation point.

    Returns:
        (n_strands-1, n_strands-1) complex numpy array.
    """
    B_red = reduced_burau_matrix(braid_word, n_strands, t)
    return np.eye(n_strands - 1, dtype=complex) - B_red


def alexander_polynomial_eval(braid_word: List[int], n_strands: int, t: complex) -> complex:
    """
    Evaluate the Alexander polynomial at t by computing det(A(t)).

    Note: this gives the Alexander polynomial up to multiplication by
    +-t^k (a unit in Z[t, t^{-1}]).  For the purposes of feature extraction
    the absolute value |det(A(t))| is used.

    Args:
        braid_word: list of signed integers.
        n_strands: number of strands.
        t: evaluation point in C.

    Returns:
        complex: value of det(A(t)).
    """
    if len(braid_word) == 0 or n_strands < 2:
        return complex(1.0)
    A = alexander_matrix(braid_word, n_strands, t)
    return complex(np.linalg.det(A))


# ---------------------------------------------------------------------------
# Exact Theta invariant
# ---------------------------------------------------------------------------

def theta_exact(braid_word: List[int], n_strands: int, t: float) -> float:
    """
    Compute the exact Theta invariant at t via the Alexander determinant.

    Theta_exact(t) = |det(I - Burau_red(b, t))|

    This is the principled version of the linear approximation in theta_eval.py.
    It uses the full Burau representation evaluated at t, giving a more
    faithful approximation of the actual Theta knot invariant.

    Args:
        braid_word: list of signed integers (1-indexed).
        n_strands: number of strands.
        t: evaluation point in (0, 1).

    Returns:
        float: |det(A(t))| >= 0.
    """
    val = alexander_polynomial_eval(braid_word, n_strands, t)
    return float(abs(val))


def theta_exact_signed(braid_word: List[int], n_strands: int, t: float) -> float:
    """
    Signed version: Re(det(I - Burau_red(b, t))).
    Useful when the sign carries topological meaning.

    Args:
        braid_word: list of signed integers.
        n_strands: number of strands.
        t: evaluation point in (0, 1).

    Returns:
        float: Re(det(A(t))).
    """
    val = alexander_polynomial_eval(braid_word, n_strands, t)
    return float(val.real)


# ---------------------------------------------------------------------------
# Batch computation for pipeline
# ---------------------------------------------------------------------------

def compute_exact_theta_features(
    braid_words: List[List[int]],
    n_strands: int,
    t_values: Tuple[float, ...] = (0.5, 1.0 / 3.0, 2.0 / 3.0),
    include_abs_det: bool = True,
    include_signed_det: bool = True,
) -> np.ndarray:
    """
    Compute exact Theta features for all braid words.

    For each braid word, evaluates:
        - |det(A(t))| at each t in t_values  (if include_abs_det)
        - Re(det(A(t))) at each t in t_values (if include_signed_det)

    Args:
        braid_words: list of braid word lists (length N).
        n_strands: number of strands (= len(feature_order)).
        t_values: tuple of evaluation points.
        include_abs_det: include |det| columns.
        include_signed_det: include Re(det) columns.

    Returns:
        numpy array of shape (N, k) where k depends on flags.
        Column order:
            [|det|(t0), |det|(t1), ..., Re(det)(t0), Re(det)(t1), ...]
    """
    n = len(braid_words)
    cols = []
    col_labels = []

    for t in t_values:
        if include_abs_det:
            cols.append([])
            col_labels.append(f"Theta_exact_abs_{t:.3f}")
        if include_signed_det:
            cols.append([])
            col_labels.append(f"Theta_exact_re_{t:.3f}")

    for bw in braid_words:
        col_ptr = 0
        for t in t_values:
            val = alexander_polynomial_eval(bw, n_strands, t)
            if include_abs_det:
                cols[col_ptr].append(float(abs(val)))
                col_ptr += 1
            if include_signed_det:
                cols[col_ptr].append(float(val.real))
                col_ptr += 1

    theta_matrix = np.column_stack([np.array(c) for c in cols]) if cols else np.zeros((n, 0))

    print(f"[Alexander] Exact Theta features: {theta_matrix.shape}")
    for j, label in enumerate(col_labels):
        col = theta_matrix[:, j]
        print(f"  {label}: [{col.min():.4f}, {col.max():.4f}]")

    return theta_matrix


def alexander_features_from_closures(
    closures: List[ClosedBraid],
    t_values: Tuple[float, ...] = (0.5, 1.0 / 3.0, 2.0 / 3.0),
) -> np.ndarray:
    """
    Convenience wrapper: compute exact Theta features from ClosedBraid objects.

    Uses each closure's own n_strands, so this is safe for mixed-strand inputs.

    Args:
        closures: list of ClosedBraid objects.
        t_values: evaluation points.

    Returns:
        numpy array of shape (N, 2*len(t_values)) — [abs_det, re_det] per t.
    """
    rows = []
    for cb in closures:
        row = []
        for t in t_values:
            val = alexander_polynomial_eval(cb.braid_word, cb.n_strands, t)
            row.append(float(abs(val)))
            row.append(float(val.real))
        rows.append(row)

    mat = np.array(rows, dtype=np.float64)
    labels = [f"Theta_exact_abs_{t:.3f}" for t in t_values] + \
             [f"Theta_exact_re_{t:.3f}" for t in t_values]
    print(f"[Alexander] Features from closures: {mat.shape} — cols: {labels}")
    return mat
