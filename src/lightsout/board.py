from __future__ import annotations

import numpy as np


class BoardState:
    def __init__(self, n: int, state: np.ndarray | None = None):
        self.n = n
        if state is None:
            self.state = np.zeros((n, n), dtype=bool)
        else:
            assert state.shape == (n, n)
            self.state = state.astype(bool, copy=True)

    def copy(self) -> "BoardState":
        return BoardState(self.n, self.state.copy())

    def to_flat(self) -> np.ndarray:
        return self.state.reshape(-1)

    @staticmethod
    def from_flat(n: int, flat: np.ndarray) -> "BoardState":
        return BoardState(n, flat.reshape(n, n))

    def count_on(self) -> int:
        return int(self.state.sum())

    def __repr__(self):
        return f"BoardState(n={self.n}, on={self.count_on()})"

    def __str__(self) -> str:
        return "\n".join(
            "".join("1" if cell else "0" for cell in row) for row in self.state
        )
