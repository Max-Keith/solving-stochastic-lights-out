from __future__ import annotations

import numpy as np


def manhattan_open(n, i, j):
    r1, c1 = divmod(i, n)
    r2, c2 = divmod(j, n)
    return abs(r1 - r2) + abs(c1 - c2)


def make_exp_kernel(n: int, alpha: float, lam: float) -> np.ndarray:
    """Return P of shape (N, N): P[i, j] = Pr(cell j flips | press i).
    Target cell flips with probability 1. Others: alpha * exp(-d/lam).
    """
    N = n * n
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                P[i, j] = 1.0
            else:
                d = (
                    manhattan_open(n, i, j) - 1
                )  # subtract 1 to make adjacent cells distance 0
                if lam <= 0:
                    p = 0.0
                else:
                    p = alpha * float(np.exp(-d / lam))
                P[i, j] = max(0.0, min(1.0, p))
    return P


def make_deterministic_kernel(n: int) -> np.ndarray:
    """Return P of shape (N, N): P[i, j] = Pr(cell j flips | press i).
    Target cell flips with probability 1. Directly adjacent cells flip with probability 1. Others: 0.
    """
    N = n * n
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                P[i, j] = 1.0
            else:
                d = manhattan_open(n, i, j)
                if d == 1:
                    p = 1.0
                else:
                    p = 0.0
                P[i, j] = max(0.0, min(1.0, p))
    return P
