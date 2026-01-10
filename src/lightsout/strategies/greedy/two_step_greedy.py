from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ...board import BoardState
from ..base import Strategy


class TwoStepExpectedGreedy(Strategy):
    """Two-step lookahead minimizing expected lights after two moves."""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        tie_break: str = "first",
    ):
        self.rng = rng or np.random.default_rng()
        self.tie_break = tie_break

        # Set at reset:
        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[NDArray] = None

    def reset(self, n: int, params: dict | None = None) -> None:
        self.n = int(n)
        self.N = n * n
        self.P = None

        if params is not None:
            P = params.get("P", None)
            if P is not None:
                P = np.asarray(P, dtype=float)
                if P.shape != (self.N, self.N):
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

    def _expected_next_state(
        self, on_probs: np.ndarray, action: int
    ) -> np.ndarray:
        """Compute expected state after pressing an action."""
        if self.P is None:
            raise ValueError("Strategy not initialized properly.")
        flip_probs = self.P[action]
        return on_probs + flip_probs * (1.0 - 2.0 * on_probs)

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.P is None or self.N is None:
            raise ValueError("Strategy not initialized properly.")

        current_probs = state.to_flat().astype(np.float64)

        two_step_values = np.empty(self.N, dtype=float)

        for first_action in range(self.N):
            next_probs = self._expected_next_state(current_probs, first_action)
            next_weight_vector = 1.0 - 2.0 * next_probs
            second_step_scores = np.dot(self.P, next_weight_vector)
            best_second_step = float(second_step_scores.min())
            two_step_values[first_action] = (
                float(next_probs.sum()) + best_second_step
            )

        if self.tie_break == "first":
            best_action = int(np.argmin(two_step_values))
        else:
            min_value = two_step_values.min()
            candidates = np.flatnonzero(
                np.isclose(two_step_values, min_value, rtol=1e-9, atol=1e-12)
            )
            best_action = int(self.rng.choice(candidates))

        return best_action
