from __future__ import annotations


def success_within_budget(count_on_history, budget_T: int) -> int:
    return int(any(c == 0 for c in count_on_history[: budget_T + 1]))


def presses_used(count_on_history) -> int:
    # number of steps taken (length-1 if solved before budget)
    return len(count_on_history) - 1


def rescue_break(det_solvable: int, solved: int):
    # returns (rescue, break)
    if det_solvable == 0 and solved == 1:
        return 1, 0
    if det_solvable == 1 and solved == 0:
        return 0, 1
    return 0, 0
