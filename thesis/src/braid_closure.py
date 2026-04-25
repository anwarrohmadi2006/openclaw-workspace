"""
Priority 1 — Braid Closure
Implement trace closure of a braid word, turning an open braid into a closed
braid (knot/link diagram). This is the topological object on which invariants
such as the Alexander polynomial and Theta invariant are properly defined.

Background:
    Given a braid b on n strands, its *trace closure* b̂ connects the top of
    the i-th strand to the bottom of the i-th strand for each i=1,...,n.
    The resulting closed braid represents a knot or link in S^3.

Reference:
    Alexander's theorem: every knot/link is isotopic to the closure of some
    braid (Alexander, 1923).  Markov's theorem gives the equivalence moves.

API overview:
    ClosedBraid         — dataclass holding braid word + metadata
    braid_closure()     — construct ClosedBraid from braid word + n_strands
    closure_components()— compute connected components (number of link components)
    closure_writhe()    — writhe of the closed braid diagram
    markov_stabilize()  — positive Markov stabilization (for ablation)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClosedBraid:
    """
    Represents a closed braid (knot/link diagram).

    Attributes:
        braid_word:  list of signed integers — generators σ_i (positive)
                     or σ_i^{-1} (negative), 1-indexed.
        n_strands:   number of braid strands.
        n_components: number of connected components of the closure
                      (= number of link components).  1 → knot.
        writhe:      algebraic crossing number (sum of signs).
        is_knot:     True iff n_components == 1.
        metadata:    optional dict for storing extra information.
    """
    braid_word: List[int]
    n_strands: int
    n_components: int = field(init=False)
    writhe: int = field(init=False)
    is_knot: bool = field(init=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.writhe = closure_writhe(self.braid_word)
        self.n_components = closure_components(self.braid_word, self.n_strands)
        self.is_knot = (self.n_components == 1)

    def __repr__(self):
        kind = "knot" if self.is_knot else f"link ({self.n_components} components)"
        return (
            f"ClosedBraid(n_strands={self.n_strands}, "
            f"length={len(self.braid_word)}, "
            f"writhe={self.writhe}, {kind})"
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def braid_closure(braid_word: List[int], n_strands: int) -> ClosedBraid:
    """
    Construct the trace closure of a braid word.

    The trace closure identifies the top and bottom endpoints of each strand:
        strand i_top  ↔  strand i_bottom   for i = 1, ..., n_strands.

    Args:
        braid_word:  list of signed integers (1-indexed generators).
                     Positive g means σ_g (positive crossing).
                     Negative g means σ_{|g|}^{-1} (negative crossing).
        n_strands:   number of braid strands.

    Returns:
        ClosedBraid instance with computed metadata.

    Raises:
        ValueError: if any generator index is out of range [1, n_strands-1].
    """
    if n_strands < 2:
        raise ValueError(f"n_strands must be >= 2, got {n_strands}")

    for g in braid_word:
        idx = abs(g)
        if idx < 1 or idx > n_strands - 1:
            raise ValueError(
                f"Generator {g} out of range for {n_strands} strands. "
                f"Valid indices: 1..{n_strands - 1}."
            )

    return ClosedBraid(braid_word=list(braid_word), n_strands=n_strands)


def closure_components(braid_word: List[int], n_strands: int) -> int:
    """
    Compute the number of connected components of the trace closure.

    Algorithm:
        Simulate the permutation induced by the braid word, then count
        cycles in the permutation (each cycle = one link component).

        σ_i acts by transposing positions i-1 and i (0-indexed).

    Args:
        braid_word:  list of signed integers (1-indexed).
        n_strands:   number of strands.

    Returns:
        int: number of link components (≥ 1).

    Example:
        braid_word=[1, 1, 1], n_strands=2  → closure = trefoil knot → 1 component
        braid_word=[1, -1],   n_strands=2  → closure = 2-component unlink → 2 components
    """
    # Build permutation from braid word (σ_i = transposition of i-1 and i)
    perm = list(range(n_strands))
    for g in braid_word:
        i = abs(g) - 1  # 0-indexed position
        perm[i], perm[i + 1] = perm[i + 1], perm[i]

    # Count cycles in permutation using union-find
    # The closure identifies strand top j with strand bottom perm[j]
    # A cycle in (j → perm[j]) = one connected component
    visited = [False] * n_strands
    n_cycles = 0
    for start in range(n_strands):
        if not visited[start]:
            n_cycles += 1
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cur = perm[cur]

    return n_cycles


def closure_writhe(braid_word: List[int]) -> int:
    """
    Compute writhe of the closed braid diagram.

    Writhe = algebraic crossing number = Σ sign(g) for g in braid_word.
    Positive crossings contribute +1, negative crossings contribute -1.

    Args:
        braid_word: list of signed integers.

    Returns:
        int: writhe value.
    """
    return sum(1 if g > 0 else -1 for g in braid_word if g != 0)


# ---------------------------------------------------------------------------
# Markov stabilization (for ablation / normalization)
# ---------------------------------------------------------------------------

def markov_stabilize(braid_word: List[int], n_strands: int, positive: bool = True) -> tuple:
    """
    Apply one Markov stabilization to the braid word.

    Markov's theorem states that two braids represent the same link iff they
    are related by braid isotopy and Markov moves.  A (positive) Markov
    stabilization appends σ_n to the word and increases the strand count by 1.

    This is used in ablation studies to verify closure-invariance of features:
    the resulting ClosedBraid should represent the same link.

    Args:
        braid_word:  original braid word (list of signed ints).
        n_strands:   current number of strands.
        positive:    if True, append +n_strands; else append -n_strands.

    Returns:
        (new_braid_word, new_n_strands) — ready to pass into braid_closure().
    """
    sign = 1 if positive else -1
    new_word = list(braid_word) + [sign * n_strands]
    return new_word, n_strands + 1


# ---------------------------------------------------------------------------
# Batch utilities for pipeline integration
# ---------------------------------------------------------------------------

def make_closures_from_braid_words(
    braid_words: List[List[int]],
    n_strands: int,
    verbose: bool = True,
) -> List[ClosedBraid]:
    """
    Construct ClosedBraid objects for a batch of braid words.

    Args:
        braid_words:  list of braid word lists (output of generate_braid_words)
        n_strands:    number of braid strands (= len(feature_order))
        verbose:      print summary statistics

    Returns:
        list of ClosedBraid objects
    """
    closures = []
    for bw in braid_words:
        if len(bw) == 0:
            # Empty braid word → identity braid → closure is n_strands-component unlink
            cb = ClosedBraid(braid_word=[], n_strands=n_strands)
        else:
            cb = braid_closure(bw, n_strands)
        closures.append(cb)

    if verbose:
        writhes = [cb.writhe for cb in closures]
        n_comps = [cb.n_components for cb in closures]
        knot_frac = np.mean([cb.is_knot for cb in closures])
        print(f"[Closure] {len(closures)} closed braids on {n_strands} strands")
        print(f"  Writhe: mean={np.mean(writhes):.2f}, range=[{min(writhes)}, {max(writhes)}]")
        print(f"  Components: mean={np.mean(n_comps):.2f}, range=[{min(n_comps)}, {max(n_comps)}]")
        print(f"  Knot fraction (1-component): {knot_frac:.3f}")

    return closures


def closure_feature_matrix(closures: List[ClosedBraid]) -> np.ndarray:
    """
    Extract a numeric feature matrix from a list of ClosedBraid objects.

    Columns:
        0  writhe         — algebraic crossing number
        1  n_components   — number of link components
        2  braid_length   — number of generators in the word
        3  n_strands      — number of strands
        4  writhe_density — writhe / braid_length  (0 if length=0)

    Args:
        closures: list of ClosedBraid objects

    Returns:
        numpy array of shape (N, 5)
    """
    rows = []
    for cb in closures:
        length = len(cb.braid_word)
        density = cb.writhe / length if length > 0 else 0.0
        rows.append([
            cb.writhe,
            cb.n_components,
            length,
            cb.n_strands,
            density,
        ])
    mat = np.array(rows, dtype=np.float64)
    print(
        f"[Closure] Feature matrix: {mat.shape} — "
        "[writhe, n_components, braid_length, n_strands, writhe_density]"
    )
    return mat
