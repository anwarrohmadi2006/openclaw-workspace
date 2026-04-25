"""
Braid Group Utilities
- Braid closure construction
- Validasi relasi braid group
- Konversi antara representasi

Referensi:
    - Artin (1925): Braid group relations
    - Alexander (1923): Alexander's theorem (every link = closure of a braid)
    - Bar-Natan & van der Veen (2025), arXiv:2509.18456
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Braid Closure
# ─────────────────────────────────────────────────────────────────────────────

class BraidClosure:
    """
    Representasi closure dari braid word.

    Braid closure: ujung atas strand ke-i dihubungkan ke ujung bawah
    strand ke-i, membentuk knot atau link.

    Referensi: Alexander's theorem — setiap knot/link bisa direpresentasikan
    sebagai closure dari suatu braid.
    """

    def __init__(self, braid_word: List[int], n_strands: int):
        """
        Args:
            braid_word: list signed int generator
            n_strands: jumlah strand
        """
        self.braid_word = braid_word
        self.n_strands = n_strands
        self._validate_generators()

    def _validate_generators(self):
        """Pastikan semua generator dalam range valid."""
        for g in self.braid_word:
            if abs(g) < 1 or abs(g) >= self.n_strands:
                raise ValueError(
                    f"Generator {g} out of range [1, {self.n_strands - 1}] "
                    f"for {self.n_strands} strands"
                )

    @property
    def writhe(self) -> int:
        """Writhe = sum of signs of all generators."""
        return sum(1 if g > 0 else -1 for g in self.braid_word)

    @property
    def length(self) -> int:
        """Panjang braid word (jumlah crossing)."""
        return len(self.braid_word)

    @property
    def n_components(self) -> int:
        """
        Hitung jumlah komponen link dari closure.
        Dengan melacak permutasi strand.
        """
        perm = list(range(self.n_strands))
        for g in self.braid_word:
            i = abs(g) - 1
            perm[i], perm[i + 1] = perm[i + 1], perm[i]

        # Hitung cycle length di permutasi
        visited = [False] * self.n_strands
        n_cycles = 0
        for start in range(self.n_strands):
            if not visited[start]:
                n_cycles += 1
                curr = start
                while not visited[curr]:
                    visited[curr] = True
                    curr = perm[curr]
        return n_cycles

    def is_knot(self) -> bool:
        """True jika closure adalah knot (1 komponen)."""
        return self.n_components == 1

    def __repr__(self):
        return (f"BraidClosure(word={self.braid_word[:5]}{'...' if len(self.braid_word) > 5 else ''}, "
                f"n_strands={self.n_strands}, writhe={self.writhe}, "
                f"n_components={self.n_components})")


def braid_word_to_closure(braid_word: List[int], n_strands: int) -> BraidClosure:
    """Buat BraidClosure dari braid word."""
    return BraidClosure(braid_word, n_strands)


# ─────────────────────────────────────────────────────────────────────────────
# Validasi Relasi Braid Group (Artin Relations)
# ─────────────────────────────────────────────────────────────────────────────

def check_braid_relation(word: List[int], n_strands: int,
                          verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Verifikasi apakah braid word COULD satisfy braid group relations.

    Catatan penting: Braid word yang dihasilkan dari bubble sort data tabular
    adalah VALID secara definisi (setiap permutation punya braid word unik via
    bubble sort), tapi kita bisa verifikasi bahwa generator-generator yang muncul
    sesuai dengan relasi Artin.

    Relasi Artin (braid group B_n):
        1. Braid relation:  sigma_i sigma_{i+1} sigma_i = sigma_{i+1} sigma_i sigma_{i+1}
           untuk |i - j| = 1
        2. Commutativity:   sigma_i sigma_j = sigma_j sigma_i
           untuk |i - j| >= 2

    Dalam konteks data tabular, kita validasi:
        - Semua generator dalam range [1, n_strands-1]
        - Tidak ada generator yang nilainya 0
        - Panjang braid word konsisten dengan n_strands

    Args:
        word: braid word
        n_strands: jumlah strand
        verbose: print detail

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True

    for g in word:
        if g == 0:
            warnings.append("Generator 0 tidak valid (harus >= 1)")
            is_valid = False
        elif abs(g) >= n_strands:
            warnings.append(f"Generator {g} >= n_strands={n_strands}")
            is_valid = False

    # Cek max generator vs expected dari bubble sort
    if word:
        max_gen = max(abs(g) for g in word)
        expected_max = n_strands - 1
        if max_gen > expected_max:
            warnings.append(
                f"Max generator {max_gen} > expected {expected_max} "
                f"untuk {n_strands} strands"
            )
            is_valid = False

        # Expected length dari bubble sort: <= n*(n-1)/2
        expected_max_len = n_strands * (n_strands - 1) // 2
        if len(word) > expected_max_len:
            warnings.append(
                f"Panjang braid word {len(word)} melebihi max bubble sort "
                f"{expected_max_len} untuk {n_strands} strands"
            )

    if verbose and warnings:
        for w in warnings:
            print(f"  [BraidValidation] WARNING: {w}")

    return is_valid, warnings


def validate_braid_words_batch(
    braid_words: List[List[int]],
    n_strands: int,
    sample_size: int = 100
) -> dict:
    """
    Validasi batch braid words (random sample untuk efisiensi).

    Args:
        braid_words: list of braid words
        n_strands: jumlah strand
        sample_size: berapa braid word yang dicek

    Returns:
        dict dengan statistik validasi
    """
    n = len(braid_words)
    indices = np.random.choice(n, min(sample_size, n), replace=False)

    n_valid = 0
    all_warnings = []
    lengths = [len(bw) for bw in braid_words]

    for i in indices:
        valid, warns = check_braid_relation(braid_words[i], n_strands)
        if valid:
            n_valid += 1
        all_warnings.extend(warns)

    stats = {
        "n_total": n,
        "n_sampled": len(indices),
        "n_valid_sampled": n_valid,
        "validity_rate": n_valid / len(indices) if indices.size > 0 else 0.0,
        "avg_length": float(np.mean(lengths)),
        "max_length": int(np.max(lengths)),
        "min_length": int(np.min(lengths)),
        "expected_max_length": n_strands * (n_strands - 1) // 2,
        "unique_warnings": list(set(all_warnings)),
    }

    print(f"[BraidValidation] Sampled {len(indices)}/{n} braid words")
    print(f"  Valid: {n_valid}/{len(indices)} ({stats['validity_rate']:.1%})")
    print(f"  Avg length: {stats['avg_length']:.1f} | "
          f"Max: {stats['max_length']} | "
          f"Expected max: {stats['expected_max_length']}")
    if stats["unique_warnings"]:
        for w in stats["unique_warnings"]:
            print(f"  WARNING: {w}")

    return stats


def n_strands_from_feature_order(feature_order: np.ndarray) -> int:
    """
    Tentukan jumlah strand dari feature order.
    n_strands = len(feature_order) + 1
    (strand extra untuk braid closure axis)
    """
    return len(feature_order) + 1
