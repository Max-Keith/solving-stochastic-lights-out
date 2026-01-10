from __future__ import annotations

from typing import Optional

import numpy as np

from ..board import BoardState
from .base import Strategy


class RandomClick(Strategy):
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

        self.n: Optional[int] = None
        self.N: Optional[int] = None

    def reset(self, n: int, params: dict | None = None):
        self.n = n
        self.N = n * n

    def select_action(self, state: BoardState, t: int, history):
        assert self.N is not None, "Strategy not initialized properly."

        return int(self.rng.integers(self.N))
