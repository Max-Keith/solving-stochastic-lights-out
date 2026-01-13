import numpy as np


def xor_probs(*rows):
    """
    Combine multiple probability rows using XOR rule.

    For binary events with probabilities p1, p2, ..., the probability
    that an odd number of them occur is computed.
    """
    rows = np.array(rows)
    return 0.5 - 0.5 * np.prod(1 - 2 * rows, axis=0)


def get_deterministic_probs(actions, n):
    """
    Get deterministic Lights Out flip probabilities for given actions.

    Parameters
    ----------
    actions : int or array-like
        Cell index/indices to press (0 to n²-1).
    n : int
        Board size.

    Returns
    -------
    probs : np.ndarray, shape (n, n)
        Flip probability grid (values are 0.0 or 1.0).
    """
    actions = np.atleast_1d(actions)
    N = n * n
    rows = []

    for a in actions:
        r, c = divmod(a, n)
        row = np.zeros(N, dtype=float)
        # Center cell
        row[a] = 1.0
        # Plus-shape neighbors
        if r > 0:
            row[a - n] = 1.0
        if r < n - 1:
            row[a + n] = 1.0
        if c > 0:
            row[a - 1] = 1.0
        if c < n - 1:
            row[a + 1] = 1.0
        rows.append(row)

    return xor_probs(*rows).reshape(n, n)


def get_stochastic_probs(P, actions, n):
    """
    Get stochastic flip probabilities from kernel P for given actions.

    Parameters
    ----------
    P : np.ndarray, shape (N, N) where N = n²
        Stochastic kernel matrix.
    actions : int or array-like
        Cell index/indices to press (0 to n²-1).
    n : int
        Board size.

    Returns
    -------
    probs : np.ndarray, shape (n, n)
        Flip probability grid.
    """
    actions = np.atleast_1d(actions)
    rows = [P[a] for a in actions]
    return xor_probs(*rows).reshape(n, n)


def get_entropy(probs):
    """
    Compute Shannon entropy (in bits) for flip probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Flip probabilities (0 to 1).

    Returns
    -------
    entropy : np.ndarray
        Shannon entropy in bits. Maximum is 1.0 (at p=0.5).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(
            probs * np.log2(probs + 1e-12)
            + (1 - probs) * np.log2(1 - probs + 1e-12)
        )
    return h


def get_difference(P, actions, n):
    """
    Get difference between stochastic and deterministic flip probabilities.

    Parameters
    ----------
    P : np.ndarray, shape (N, N) where N = n²
        Stochastic kernel matrix.
    actions : int or array-like
        Cell index/indices to press (0 to n²-1).
    n : int
        Board size.

    Returns
    -------
    diff : np.ndarray, shape (n, n)
        Difference grid (stochastic - deterministic).
    """
    det = get_deterministic_probs(actions, n)
    stoch = get_stochastic_probs(P, actions, n)
    return stoch - det
