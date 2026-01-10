from typing import Optional

import numpy as np
from numpy.typing import NDArray

from lightsout.board import BoardState
from lightsout.strategies.base import NoPlanError, Strategy
from lightsout.strategies.greedy import ExpectedLightsGreedy


class StandardRollout(Strategy):
    """Monte Carlo rollout that evaluates all actions independently."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.n: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[NDArray] = None
        self.depth = 2
        self.n_rollouts = 64
        self.cost = "count_on"  # or "success"
        self.default_policy: Optional[Strategy] = None

    def reset(self, n: int, params: dict | None = None):
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
            if "depth" in params:
                self.depth = int(params["depth"])
            if "n_rollouts" in params:
                self.n_rollouts = int(params["n_rollouts"])
            if "cost" in params:
                self.cost = str(params["cost"])

        self.default_policy = ExpectedLightsGreedy(rng=self.rng)

        if self.P is None:
            raise ValueError("Need to provide kernel matrix P in params.")
        self.default_policy.reset(n, params={"P": self.P, "rng": self.rng})

    def _step_stochastic(
        self, state: BoardState, action: int, rng: np.random.Generator
    ) -> BoardState:
        """Apply stochastic action to state."""
        if self.P is None:
            raise ValueError("Kernel matrix P is not initialized.")
        if self.n is None:
            raise ValueError("Board dimension 'n' not set.")

        flat_state = state.to_flat().astype(np.uint8)
        flip_probs = self.P[action]
        flips = (rng.random(self.N) < flip_probs).astype(np.uint8)
        new_state = (flat_state ^ flips).astype(bool)
        return BoardState.from_flat(self.n, new_state)

    def _default_action(self, state: BoardState) -> int:
        """Select action using default greedy policy."""
        if self.default_policy is None:
            raise ValueError("Default policy not initialized.")
        return self.default_policy.select_action(state, t=0, history=None)

    def _estimate_action_value(self, state: BoardState, action: int) -> float:
        """Estimate action value via Monte Carlo rollouts."""
        total_cost = 0.0
        for _ in range(self.n_rollouts):
            seed = int(self.rng.integers(2**63 - 1))
            rollout_rng = np.random.default_rng(seed)

            current_state = state.copy()
            current_state = self._step_stochastic(
                current_state, action, rollout_rng
            )

            remaining_steps = self.depth - 1
            while remaining_steps > 0 and current_state.count_on() > 0:
                next_action = self._default_action(current_state)
                current_state = self._step_stochastic(
                    current_state, next_action, rollout_rng
                )
                remaining_steps -= 1

            if self.cost == "count_on":
                cost = float(current_state.count_on())
            elif self.cost == "success":
                cost = 0.0 if current_state.count_on() == 0 else 1.0
            else:
                raise ValueError(f"Unknown cost function '{self.cost}'")

            total_cost += cost

        return total_cost / self.n_rollouts

    def select_action(self, state: BoardState, t: int, history) -> int:
        if self.P is None or self.n is None or self.N is None:
            raise NoPlanError("Strategy not initialized properly.")

        # Evaluate ALL actions
        best_action = None
        best_value = float("inf")

        for action in range(self.N):
            value = self._estimate_action_value(state, action)

            # Tie-break by choosing the first action
            if value < best_value or (
                value == best_value
                and (best_action is None or action < best_action)
            ):
                best_action = action
                best_value = value

        assert best_action is not None
        return best_action
