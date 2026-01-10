from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ...board import BoardState
from ..base import Strategy


class ExpectedLightsGreedy(Strategy):
    """Pick action that minimizes expected number of lights after one press."""

    def __init__(
        self, rng: np.random.Generator | None = None, tie_break: str = "first"
    ):
        self.rng = rng or np.random.default_rng()
        self.tie_break = tie_break

        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[NDArray[np.float64]] = None

    def reset(self, n: int, params: dict | None = None):
        self.n = int(n)
        self.N = n * n

        self.P = None

        if params is not None:
            P = params.get("P", None)
            if P is not None:
                P = np.asarray(P, dtype=float)
                if P.shape != (
                    self.N,
                    self.N,
                ):
                    raise ValueError(
                        f"Expected P of shape {(self.N, self.N)}, got {P.shape}"
                    )
                self.P = P

            tb = params.get("tie_break", None)
            if tb in ("first", "random"):
                self.tie_break = tb

            rng = params.get("rng", None)
            if isinstance(rng, np.random.Generator):
                self.rng = rng

        if self.P is None:
            raise ValueError("Need to provide kernel matrix P in params.")

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.P is None:
            raise ValueError(
                "ExpectedParityGreedy.select_action called before reset with kernel matrix."
            )

        # Current state as 0/1 vector
        flat_state = state.to_flat().astype(np.int8)

        # +1 for OFF, -1 for ON
        state_weight_vector = 1 - 2 * flat_state

        scores = np.dot(self.P, state_weight_vector)

        if self.tie_break == "first":
            best_action = np.argmin(scores)
        else:
            min_score = np.min(scores)
            candidates = np.flatnonzero(scores == min_score)
            best_action = self.rng.choice(candidates)

        return int(best_action)
