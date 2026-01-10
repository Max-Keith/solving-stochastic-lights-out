from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import numpy as np


def _neighbors_open(n, r, c):
    neigh = [(r, c)]
    if r > 0:
        neigh.append((r - 1, c))
    if r < n - 1:
        neigh.append((r + 1, c))
    if c > 0:
        neigh.append((r, c - 1))
    if c < n - 1:
        neigh.append((r, c + 1))
    return neigh


def build_A(n: int) -> np.ndarray:
    """Return the N×N effect matrix A over GF(2) for Lights Out.
    Column j encodes the cells toggled when pressing cell j.
    """
    N = n * n
    A = np.zeros((N, N), dtype=np.uint8)  # use 0/1 ints for XOR via mod2

    def idx(r, c):
        return r * n + c

    for r in range(n):
        for c in range(n):
            j = idx(r, c)
            for rr, cc in _neighbors_open(n, r, c):
                A[idx(rr, cc), j] = 1
    return A


def gf2_rref_augmented(
    A: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, list[int]]:
    """Return RREF of augmented matrix [A|b] over GF(2) and list of pivot columns."""
    A = (A % 2).astype(np.uint8)
    b = (b % 2).astype(np.uint8).reshape(-1, 1)
    m, n = A.shape
    M = np.concatenate([A.copy(), b.copy()], axis=1)  # shape (m, n+1)

    row = 0
    pivcols: list[int] = []
    for col in range(n):
        # find a pivot in/under current row
        pivot = None
        for r in range(row, m):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        # swap pivot row up
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        # eliminate ALL other rows (Gauss-Jordan)
        for r in range(m):
            if r != row and M[r, col]:
                M[r, :] ^= M[row, :]
        pivcols.append(col)
        row += 1
        if row == m:
            break
    return M, pivcols


def gf2_solve_with_nullspace(
    A: np.ndarray, b: np.ndarray
) -> Tuple[Optional[np.ndarray], List[np.ndarray], bool]:
    """Solve A x = b over GF(2), and return a nullspace basis of A.

    Returns:
        x0: one particular solution (length n, uint8) or None if inconsistent
        basis: list of nullspace basis vectors v (length n, uint8) with A v = 0
        solvable: bool
    """
    m, n = A.shape
    R, pivcols = gf2_rref_augmented(A, b)  # R is [RREF(A) | r]
    R_A = R[:, :n]
    R_b = R[:, n]

    # Inconsistency check: 0...0 | 1 rows
    if np.any((R_A.sum(axis=1) == 0) & (R_b == 1)):
        return None, [], False

    # Particular solution: set free vars = 0, solve pivot vars from RREF
    x0 = np.zeros((n,), dtype=np.uint8)
    # rows with pivots correspond 1-to-1 to pivcols in this construction
    for ri, pc in enumerate(pivcols):
        # Row ri: R_A[ri, pc] = 1; equation: x_pc = R_b[ri] ^ sum_{j>pc} R_A[ri, j]*x_j
        rhs = R_b[ri]
        if pc + 1 < n:
            rhs ^= int(
                np.bitwise_and(R_A[ri, pc + 1 :], x0[pc + 1 :]).sum() % 2
            )
        x0[pc] = rhs

    # Nullspace basis: for each free column f, set x_f=1, others free=0; solve pivot vars
    frees = [j for j in range(n) if j not in pivcols]
    basis: list[np.ndarray] = []
    for f in frees:
        v = np.zeros((n,), dtype=np.uint8)
        v[f] = 1
        for ri, pc in enumerate(pivcols):
            rhs = 0
            if pc + 1 < n:
                rhs ^= int(
                    np.bitwise_and(R_A[ri, pc + 1 :], v[pc + 1 :]).sum() % 2
                )
            v[pc] = rhs
        basis.append(v)

    return x0, basis, True


def gf2_min_weight_solution(
    A: np.ndarray, b: np.ndarray
) -> Tuple[Optional[np.ndarray], bool]:
    """Return the minimum-Hamming-weight solution to A x = b (if solvable)."""
    x0, basis, ok = gf2_solve_with_nullspace(A, b)
    if not ok or x0 is None:
        return None, False
    # Try all combinations of nullspace basis vectors
    best = x0.copy()
    best_w = int(best.sum())
    k = len(basis)
    for r in range(1, k + 1):
        for combo in itertools.combinations(range(k), r):
            cand = x0.copy()
            for idx in combo:
                cand ^= basis[idx]
            w = int(cand.sum())
            if w < best_w:
                best, best_w = cand, w
    return best, True
