from __future__ import annotations
from typing import Protocol
from ..board import BoardState


class NoPlanError(Exception):
    """Raised by a strategy when no valid plan exists for the given state."""

    pass


class Strategy(Protocol):
    def reset(self, n: int, params: dict | None = None): ...
    def select_action(self, state: BoardState, t: int, history) -> int: ...
