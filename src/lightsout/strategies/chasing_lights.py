from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np

from ..board import BoardState
from .base import NoPlanError, Strategy


def _press_in_place(grid: np.ndarray, r: int, c: int) -> None:
    """Toggle cell and its neighbors."""
    n = grid.shape[0]
    grid[r, c] ^= True
    if r > 0:
        grid[r - 1, c] ^= True
    if r < n - 1:
        grid[r + 1, c] ^= True
    if c > 0:
        grid[r, c - 1] ^= True
    if c < n - 1:
        grid[r, c + 1] ^= True


def _simulate_with_first_row(
    init: np.ndarray, first_row_mask: int
) -> Tuple[np.ndarray, bool]:
    """Apply chasing-lights with given first-row presses."""
    n = init.shape[0]
    grid = init.copy()
    presses = np.zeros_like(grid, dtype=bool)

    for c in range(n):
        if (first_row_mask >> c) & 1:
            presses[0, c] = True
            _press_in_place(grid, 0, c)

    for r in range(1, n):
        for c in range(n):
            if grid[r - 1, c]:
                presses[r, c] = True
                _press_in_place(grid, r, c)

    is_valid = not grid[n - 1].any()
    return presses, is_valid


class ChasingLights(Strategy):
    """
    Try all possible first-row presses and chase lights down.
    Finds the minimum-weight solution if one exists.
    """

    def __init__(
        self, tie_break: Literal["min_weight", "lexicographic"] = "min_weight"
    ):
        self.plan: Optional[List[int]] = None
        self.tie_break: Literal["min_weight", "lexicographic"] = tie_break

        self.n: Optional[int] = None
        self.N: Optional[int] = None

    def reset(self, n: int, params: dict | None = None) -> None:
        self.n = n
        self.N = n * n
        self.plan = None
        if params and "tie_break" in params:
            tb = params["tie_break"]
            if tb in ("min_weight", "lexicographic"):
                self.tie_break = tb

    def _candidate_key(
        self,
        n: int,
        first_row_mask: int,
        presses: np.ndarray,
    ) -> tuple:
        """Build sorting key for tie-breaking."""
        if self.tie_break == "min_weight":
            return (
                int(presses.sum()),
                tuple(int(x) for x in presses.reshape(-1)),
            )
        else:
            first_row_bits = tuple((first_row_mask >> c) & 1 for c in range(n))
            return (first_row_bits, tuple(int(x) for x in presses.reshape(-1)))

    def _compute_plan(self, state: BoardState) -> list[int]:
        assert (
            self.n is not None and self.N is not None
        ), "Strategy not initialized properly."

        n = self.n
        initial_grid = state.state.copy()

        best_key: Optional[tuple] = None
        best_plan: Optional[np.ndarray] = None

        for mask in range(1 << n):
            presses, is_valid = _simulate_with_first_row(initial_grid, mask)
            if not is_valid:
                continue

            candidate_key = self._candidate_key(n, mask, presses)

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_plan = presses

        if best_plan is None:
            self.plan = []
            return self.plan

        flat_plan = best_plan.reshape(-1)
        self.plan = [i for i in range(self.N) if flat_plan[i]]
        return self.plan

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.plan is None:
            self._compute_plan(state)
        if self.plan is None or len(self.plan) == 0:
            raise NoPlanError("No valid chasing-lights plan for this board.")

        return int(self.plan.pop(0))
