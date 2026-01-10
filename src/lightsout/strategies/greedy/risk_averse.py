from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ...board import BoardState
from ..base import Strategy


class RiskAverseGreedy(Strategy):
    """Minimize expected lights plus a penalty for variance."""

    def __init__(
        self,
        beta: float = 0.5,
        tie_break: str = "first",
        rng: Optional[np.random.Generator] = None,
    ):
        self.beta = float(beta)
        self.tie_break = tie_break
        self.rng = rng or np.random.default_rng()
        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[NDArray] = None

    def reset(self, n: int, params: dict | None = None):
        self.n = int(n)
        self.N = n * n
        self.P = None

        if params is not None:
            P = params.get("P")
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

            beta = params.get("beta", None)
            if beta is not None:
                self.beta = float(beta)

        if self.P is None:
            raise ValueError("Need to provide kernel matrix P in params.")

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.P is None:
            raise ValueError("Strategy not initialized properly.")

        # Current state as 0/1 float vector
        flat_state = state.to_flat().astype(float)

        # +1 for OFF, -1 for ON
        state_weight_vector = 1.0 - 2.0 * flat_state

        # on_prob_after_press[a, j] = probability cell j is ON after pressing a
        on_prob_after_press = flat_state + self.P * state_weight_vector[None, :]

        # Expected number of ON cells and their variance for each action
        expected_lights = on_prob_after_press.sum(axis=1)
        variance_lights = (
            on_prob_after_press * (1.0 - on_prob_after_press)
        ).sum(axis=1)

        scores = expected_lights + self.beta * variance_lights

        if self.tie_break == "first":
            best_action = int(np.argmin(scores))
        else:
            min_score = np.min(scores)
            candidates = np.flatnonzero(scores == min_score)
            best_action = int(self.rng.choice(candidates))

        return best_action
