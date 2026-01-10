from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ...board import BoardState
from ..base import Strategy


class ProbAtMostKGreedy(Strategy):
    """
    Greedy strategy that picks actions based on the probability of having
    at most K lights on after the next press. Uses a Poisson-binomial
    distribution to compute this probability.
    """

    def __init__(
        self,
        max_on: int = 0,
        rng: np.random.Generator | None = None,
        tie_break: str = "first",
    ):
        self.raw_max_on = int(max_on)
        self.rng = rng or np.random.default_rng()
        self.tie_break = tie_break

        # Set in reset:
        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[NDArray] = None
        self.K: Optional[int] = None

    def reset(self, n: int, params: dict | None = None):
        self.n = int(n)
        self.N = n * n
        self.P = None

        max_on = self.raw_max_on
        if params is not None:
            P = params.get("P", None)
            if P is not None:
                P = np.asarray(P, dtype=float)
                if P.shape != (self.N, self.N):
                    raise ValueError(
                        f"Expected P of shape {(self.N, self.N)}, got {P.shape}"
                    )
                self.P = P

            if "max_on" in params:
                max_on = int(params["max_on"])

            tb = params.get("tie_break", None)
            if tb in ("first", "random"):
                self.tie_break = tb

            rng = params.get("rng", None)
            if isinstance(rng, np.random.Generator):
                self.rng = rng

        if self.P is None:
            raise ValueError("Need to provide kernel matrix P in params.")

        max_on = max(0, max_on)
        self.K = min(max_on, self.N)

    def _prob_at_most_K(self, on_probs: np.ndarray, threshold: int) -> float:
        """Compute probability of having at most K lights on using DP."""
        # dp[k] = prob of exactly k lights on
        dp = np.zeros(threshold + 1, dtype=float)
        dp[0] = 1.0

        for prob_on in on_probs:
            new_dp = dp * (1.0 - prob_on)
            if threshold >= 1:
                new_dp[1:] += dp[:-1] * prob_on
            dp = new_dp

        return float(dp.sum())

    def select_action(self, state: BoardState, t: int, history) -> int:
        assert (
            self.P is not None and self.N is not None and self.K is not None
        ), "Strategy not initialized properly."

        flat_state = state.to_flat().astype(np.int8)
        threshold = self.K

        scores = np.empty(self.N, dtype=float)

        for action in range(self.N):
            flip_probs = self.P[action]
            on_probs_after_press = np.where(
                flat_state == 1, 1.0 - flip_probs, flip_probs
            )
            scores[action] = self._prob_at_most_K(
                on_probs_after_press, threshold
            )

        max_score = np.max(scores)
        candidates = np.flatnonzero(scores == max_score)

        if self.tie_break == "random" and len(candidates) > 1:
            best_action = int(self.rng.choice(candidates))
        else:
            best_action = int(candidates[0])

        return best_action
