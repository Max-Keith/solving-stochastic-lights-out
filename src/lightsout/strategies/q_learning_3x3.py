from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..board import BoardState
from ..rl.utils_3x3 import board_to_index_3x3
from .base import Strategy


class QLearning3x3(Strategy):
    """3x3 strategy using pre-trained Q-learning table."""

    def __init__(
        self,
        q_dir: str,
        rng: np.random.Generator | None = None,
        epsilon: float = 0.0,
    ):
        self.q_dir = q_dir  # e.g. "q_tables"
        self.rng = rng or np.random.default_rng()
        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.epsilon: float = float(epsilon)
        self.Q: Optional[NDArray] = None

    def reset(self, n: int, params: dict | None = None) -> None:
        if n != 3:
            raise ValueError(f"QLearningPolicy3x3 only supports n=3, got n={n}")

        self.n = int(n)
        self.N = n * n

        if params is not None:
            alpha = params.get("alpha", None)
            lam = params.get("lambda", None)

            fname = Path(self.q_dir) / f"q_3x3_alpha{alpha}_lambda{lam}.npy"
            self.Q = np.load(fname).astype(np.float32)

            eps = params.get("epsilon", None)
            if eps is not None:
                self.epsilon = float(eps)
            rng = params.get("rng", None)
            if isinstance(rng, np.random.Generator):
                self.rng = rng

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.n != 3:
            raise RuntimeError("QLearningPolicy3x3 not reset or wrong n.")
        if self.Q is None:
            raise RuntimeError("QLearningPolicy3x3 Q-table not loaded.")

        state_index = board_to_index_3x3(state)

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.N))

        q_values = self.Q[state_index]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(best_actions[0])
