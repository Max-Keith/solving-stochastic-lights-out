from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..algebra import build_A, gf2_min_weight_solution
from ..board import BoardState
from .base import NoPlanError, Strategy


class LinearAlgebraReplanning(Strategy):
    """
    Solve deterministic Lights Out at each step and pick one press from the solution.
    Replans after every move based on the current state.
    """

    def __init__(self, tie_break: str = "first"):
        self.tie_break = tie_break

        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.A: Optional[NDArray] = None
        self.rng = np.random.default_rng()

    def reset(self, n: int, params: dict | None = None) -> None:
        self.n = int(n)
        self.N = n * n

        if params is not None:
            tb = params.get("tie_break", None)
            if tb in ("first", "random"):
                self.tie_break = tb

            rng = params.get("rng", None)
            if isinstance(rng, np.random.Generator):
                self.rng = rng

        # Build deterministic Lights Out matrix once for this board size
        self.A = build_A(self.n)

    def _solve_current_state(self, state: BoardState) -> np.ndarray | None:
        """Find minimum-weight solution for current state."""
        assert (
            self.A is not None
        ), "LinearAlgebraReplanning: A not built, call reset() first"

        target_state = state.to_flat().astype(np.uint8)

        if target_state.sum() == 0:
            return None

        solution, is_valid = gf2_min_weight_solution(self.A, target_state)
        if not is_valid or solution is None:
            return None
        if solution.sum() == 0:
            return None
        return solution

    def select_action(self, state: BoardState, t: int, history) -> int:
        solution = self._solve_current_state(state)
        if solution is None:
            return int(self.rng.integers(0, self.N))

        candidate_actions = np.flatnonzero(solution)
        if len(candidate_actions) == 0:
            raise NoPlanError(
                "LinearAlgebraReplanning: empty plan (only zero solution)."
            )

        if self.tie_break == "random" and len(candidate_actions) > 1:
            best_action = int(self.rng.choice(candidate_actions))
        else:
            best_action = int(candidate_actions[0])

        return best_action
