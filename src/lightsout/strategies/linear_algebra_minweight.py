from __future__ import annotations

from typing import Optional

import numpy as np

from ..algebra import build_A, gf2_min_weight_solution
from ..board import BoardState
from .base import NoPlanError, Strategy


class LinearAlgebraMinWeight(Strategy):
    """Solve deterministic Lights Out for minimum-weight solution upfront."""

    def __init__(self):
        self.plan: list[int] | None = None
        self.n: Optional[int] = None
        self.N: Optional[int] = None

    def reset(self, n: int, params: dict | None = None) -> None:
        self.n = n
        self.N = n * n
        self.plan = None

    def _compute_plan(self, state: BoardState) -> None:
        assert (
            self.n is not None and self.N is not None
        ), "Strategy not initialized properly."
        system_matrix = build_A(self.n)
        target_state = state.to_flat().astype(np.uint8)
        solution, is_valid = gf2_min_weight_solution(
            system_matrix, target_state
        )
        if not is_valid or solution is None:
            self.plan = []
            return
        self.plan = [i for i in range(self.N) if solution[i] == 1]

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.plan is None:
            self._compute_plan(state)

        if self.plan is None or len(self.plan) == 0:
            raise NoPlanError(
                "No valid linear algebra solution for this board."
            )
        return int(self.plan.pop(0))
