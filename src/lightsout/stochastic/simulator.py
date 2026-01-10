from __future__ import annotations

import numpy as np

from lightsout.strategies.base import NoPlanError

from ..board import BoardState


class Simulator:
    def __init__(self, P: np.ndarray, rng: np.random.Generator | None = None):
        self.P = P  # shape (N, N)
        self.n = int(np.sqrt(P.shape[0]))
        self.N = self.n * self.n
        self.rng = rng or np.random.default_rng()

    def step(self, state: BoardState, action: int) -> BoardState:
        flat = state.to_flat().astype(np.uint8)
        probs = self.P[action]
        flips = (self.rng.random(self.N) < probs).astype(np.uint8)
        next_flat = (flat ^ flips).astype(bool)
        return BoardState.from_flat(self.n, next_flat)

    def run(self, init: BoardState, policy, T: int):
        s = init.copy()
        counts = [s.count_on()]  # start with initial
        actions = []
        for t in range(T):
            try:
                a = policy.select_action(s, t, counts)  # pass counts if needed
            except NoPlanError:
                # Stop immediately: no action taken, no additional count appended.
                break
            s = self.step(s, a)
            actions.append(a)
            counts.append(s.count_on())
            if counts[-1] == 0:
                break
        # return both, so you can plot or analyze either way
        return s, actions, counts
