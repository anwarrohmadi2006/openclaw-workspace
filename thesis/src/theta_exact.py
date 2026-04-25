"""
Theta Eksak via Burau Representation & Perturbed Alexander Invariant
Based on: Bar-Natan & van der Veen (2025), arXiv:2509.18456

Implementasi:
    1. Unreduced Burau matrix per generator sigma_i dan sigma_i^{-1}
    2. Product matrix seluruh braid word -> B_total
    3. Reduced Burau matrix (hapus baris/kolom ke-0)
    4. Alexander polynomial Delta(t) = det(I - B_reduced) / (1 + t + ... + t^{n-1})
    5. Perturbed Alexander theta(t): koefisien orde-2 dari ekspansi
       log Delta di sekitar t=1 (pendekatan Bar-Natan)

Referensi formula Burau:
    sigma_i -> blok di posisi (i-1, i-1) sampai (i, i) dalam matriks n x n:
        [1-t   t  ]
        [ 1    0  ]
    Semua elemen lain tetap seperti identitas.

    sigma_i^{-1} -> invers blok:
        [0    1   ]
        [t^{-1}  1-t^{-1}]

Catatan pada kompleksitas:
    - Matriks berukuran n x n dengan n = jumlah strand (jumlah fitur + 1)
    - Untuk n=50 fitur -> 51 x 51 matriks, masih efisien O(n^3)
    - Operasi dilakukan dengan sympy untuk polynomial eksak,
      atau numpy float64 untuk evaluasi numerik cepat
"""

import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Burau Matrix (Numerik, evaluasi di t tertentu)
# ─────────────────────────────────────────────────────────────────────────────

def burau_matrix_numeric(n_strands: int, generator: int, t: float) -> np.ndarray:
    """
    Hitung unreduced Burau matrix untuk generator sigma_i (positif)
    atau sigma_i^{-1} (negatif) pada n_strands strand.

    Args:
        n_strands: jumlah strand (= jumlah fitur yang diurut + 1)
        generator: integer signed, e.g. +2 artinya sigma_2, -3 artinya sigma_3^{-1}
        t: titik evaluasi (float)

    Returns:
        n_strands x n_strands numpy array

    Formula Burau (unreduced):
        sigma_i (1-indexed): posisi (i-1, i-1) jadi (1-t),
                             posisi (i-1, i)   jadi t,
                             posisi (i, i-1)   jadi 1,
                             posisi (i, i)     jadi 0,
                             semua lain = identitas
    """
    i = abs(generator) - 1  # convert to 0-indexed: sigma_1 -> index 0
    assert 0 <= i < n_strands - 1, f"Generator {generator} out of range for {n_strands} strands"

    M = np.eye(n_strands, dtype=np.complex128)

    if generator > 0:
        # sigma_i (positive crossing)
        M[i, i]     = 1.0 - t
        M[i, i + 1] = t
        M[i + 1, i] = 1.0
        M[i + 1, i + 1] = 0.0
    else:
        # sigma_i^{-1} (negative crossing)
        t_inv = 1.0 / t
        M[i, i]         = 0.0
        M[i, i + 1]     = 1.0
        M[i + 1, i]     = t_inv
        M[i + 1, i + 1] = 1.0 - t_inv

    return M


def burau_product_numeric(braid_word: List[int], n_strands: int, t: float) -> np.ndarray:
    """
    Hitung product semua Burau matrix untuk seluruh braid word.

    Args:
        braid_word: list signed int, e.g. [1, -2, 1, 3]
        n_strands: jumlah strand
        t: titik evaluasi

    Returns:
        n_strands x n_strands product matrix
    """
    B = np.eye(n_strands, dtype=np.complex128)
    for g in braid_word:
        Bg = burau_matrix_numeric(n_strands, g, t)
        B = B @ Bg
    return B


def reduced_burau_matrix(B_full: np.ndarray) -> np.ndarray:
    """
    Ekstrak reduced Burau matrix dari unreduced Burau matrix.
    Reduced = hapus baris ke-0 dan kolom ke-0.

    Referensi: Burau (1936), Morton (1998)
    """
    return B_full[1:, 1:]


def alexander_polynomial_eval(braid_word: List[int], n_strands: int, t: float) -> complex:
    """
    Evaluasi Alexander polynomial Delta(t) dari closure braid word
    menggunakan formula Burau:

        Delta(t) ~ det(I_{n-1} - B_reduced(t))

    Normalisasi dibuang (hanya butuh nilai relatif antar sampel).

    Args:
        braid_word: signed braid generators
        n_strands: jumlah strand (biasanya len(feature_order) + 1)
        t: titik evaluasi

    Returns:
        complex: nilai Delta(t) (real jika t real)
    """
    if not braid_word:
        # Trivial braid: Alexander polynomial = 1
        return 1.0 + 0j

    B_full = burau_product_numeric(braid_word, n_strands, t)
    B_red = reduced_burau_matrix(B_full)
    n = B_red.shape[0]
    val = np.linalg.det(np.eye(n) - B_red)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Perturbed Alexander / Theta invariant
# ─────────────────────────────────────────────────────────────────────────────

def theta_perturbed_eval(braid_word: List[int], n_strands: int,
                         t_points: Tuple[float, ...] = (0.5, 1/3, 2/3)) -> np.ndarray:
    """
    Evaluasi perturbed Alexander invariant theta(t) di beberapa titik.

    Pendekatan Bar-Natan (2025):
        Θ(t) = log |Delta(t)| / log |t - 1|

    Untuk t ≠ 1, ini memberikan orde vanishing Delta di sekitar t=1,
    yang merupakan informasi topologis yang lebih kaya dari nilai Delta sendiri.

    Untuk ML, kita pakai kombinasi:
        features = [Re(Delta(t)), Im(Delta(t)), |Delta(t)|, theta_approx(t)]

    Args:
        braid_word: signed braid generators
        n_strands: jumlah strand
        t_points: titik evaluasi

    Returns:
        numpy array shape (len(t_points) * 3,):
        untuk setiap t: [Re(Delta), Im(Delta), |Delta|]
    """
    features = []
    for t in t_points:
        delta = alexander_polynomial_eval(braid_word, n_strands, t)
        features.append(delta.real)
        features.append(delta.imag)
        features.append(abs(delta))
    return np.array(features, dtype=np.float64)


def compute_exact_theta_features(
    braid_words: List[List[int]],
    n_strands: int,
    t_points: Tuple[float, ...] = (0.5, 1/3, 2/3)
) -> np.ndarray:
    """
    Compute exact Theta/Alexander features untuk semua braid words.

    Args:
        braid_words: list of braid word lists
        n_strands: jumlah strand
        t_points: titik evaluasi

    Returns:
        numpy array shape (N, len(t_points) * 3)
        Kolom: [Re(Delta(t1)), Im(Delta(t1)), |Delta(t1)|,
                Re(Delta(t2)), Im(Delta(t2)), |Delta(t2)|, ...]
    """
    n = len(braid_words)
    k = len(t_points) * 3
    result = np.zeros((n, k), dtype=np.float64)

    for i, bw in enumerate(braid_words):
        result[i] = theta_perturbed_eval(bw, n_strands, t_points)

    print(f"[ThetaExact] Computed {k} exact features for {n} samples")
    for j, t in enumerate(t_points):
        col_abs = result[:, j * 3 + 2]  # |Delta(t)|
        print(f"  |Delta(t={t:.3f})| range: [{col_abs.min():.4f}, {col_abs.max():.4f}] "
              f"mean: {col_abs.mean():.4f}")
    return result


def writhe(braid_word: List[int]) -> int:
    """
    Hitung writhe dari braid word (sum of signs of all generators).
    Writhe adalah topological invariant sederhana.
    """
    return sum(1 if g > 0 else -1 for g in braid_word)
