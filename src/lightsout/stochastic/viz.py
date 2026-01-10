import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def xor_probs(*rows):
    rows = np.array(rows)
    return 0.5 - 0.5 * np.prod(1 - 2 * rows, axis=0)


def _combine_rows(P, actions, n):
    """
    Combine flip probabilities for a sequence of actions using XOR rule.
    If an action appears multiple times, it is applied multiple times.
    """
    rows = [P[a] for a in actions]
    return xor_probs(*rows).reshape(n, n)


def _deterministic_combined(actions, n):
    """
    Build the deterministic Lights Out (plus-shape) flip probability grid
    for one or multiple actions, then combine via XOR rule.
    If an action appears multiple times, it is applied multiple times.
    """
    N = n * n
    rows = []
    for a in actions:
        r, c = divmod(a, n)
        row = np.zeros(N, dtype=float)
        # Center
        row[a] = 1.0
        # Neighbors
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


def show_action_heatmap(
    P,
    actions,
    n: int,
    ax=None,
    pressed_color="red",
    titles=("Deterministic", "Stochastic"),
    cmap="viridis",
    include_deterministic=True,
):
    """
    Show flip probabilities for one or multiple actions.

    Modes:
      1) Provide a stochastic kernel P (shape (N,N)) and set include_deterministic=True
         (default) -> show side-by-side deterministic (computed internally) and stochastic.
      2) Provide a stochastic kernel P with include_deterministic=False -> single heatmap.
      3) (Backward compatible) Provide a tuple (P_det, P_stoch) to explicitly show both.

    Parameters
    ----------
    P : np.ndarray or (np.ndarray, np.ndarray)
        Stochastic kernel or (deterministic_kernel, stochastic_kernel) tuple.
    actions : int or iterable[int]
        One or more pressed cell indices. Combined via XOR rule.
    n : int
        Board size.
    """
    actions = np.atleast_1d(actions)

    # Backward compatible tuple branch
    if isinstance(P, tuple) and len(P) == 2:
        P_det, P_stoch = P
        det_data = _combine_rows(P_det, actions, n)
        stoch_data = _combine_rows(P_stoch, actions, n)
        data_list = [(det_data, titles[0]), (stoch_data, titles[1])]
    else:
        # Compute deterministic on the fly if requested
        if include_deterministic:
            det_data = _deterministic_combined(actions, n)
            stoch_data = _combine_rows(P, actions, n)
            data_list = [(det_data, titles[0]), (stoch_data, titles[1])]
        else:
            # Single heatmap path
            if ax is None:
                _, ax = plt.subplots(figsize=(3.5, 3.5))
            combo = _combine_rows(P, actions, n)
            im = ax.imshow(combo, cmap=cmap, vmin=0.0, vmax=1.0)
            for action in actions:
                r, c = divmod(action, n)
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        edgecolor=pressed_color,
                        facecolor="none",
                        linewidth=2,
                    )
                )
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pr(flip)")
            return ax

    # Side-by-side rendering
    _, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    out_axes = []
    for ax_i, (data, title) in zip(axes, data_list):
        im = ax_i.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
        for action in actions:
            r, c = divmod(action, n)
            ax_i.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    edgecolor=pressed_color,
                    facecolor="none",
                    linewidth=2,
                )
            )
        ax_i.set_xticks(range(n))
        ax_i.set_yticks(range(n))
        ax_i.set_xlabel("col")
        ax_i.set_ylabel("row")
        ax_i.set_title(title)
        plt.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04, label="Pr(flip)")
        out_axes.append(ax_i)
    return out_axes


def show_difference_heatmap(P, actions, n, cmap="coolwarm", centered=True):
    """
    Deterministic vs stochastic difference: (stochastic - deterministic).
    Positive => stochastic more likely to flip.
    """
    actions = np.atleast_1d(actions)
    det = _deterministic_combined(actions, n)
    stoch = _combine_rows(P, actions, n)
    diff = stoch - det
    v = np.abs(diff).max() if centered else None
    _, ax = plt.subplots(figsize=(3.5, 3.5))
    im = ax.imshow(
        diff,
        cmap=cmap,
        vmin=(-v if centered and v is not None else None),
        vmax=(v if centered and v is not None else None),
    )
    for a in actions:
        r, c = divmod(a, n)
        ax.add_patch(
            Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                edgecolor="black",
                facecolor="none",
                linewidth=1.5,
            )
        )
    ax.set_title("Stochastic - Deterministic")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ Pr(flip)")
    return ax


def show_entropy_heatmap(P, actions, n, cmap="magma"):
    """
    Shannon entropy (bits) of flip probability for combined actions.
    Highlights uncertainty (0 at p=0/1, peak at p=0.5).
    """
    actions = np.atleast_1d(actions)
    probs = _combine_rows(P, actions, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(
            probs * np.log2(probs + 1e-12)
            + (1 - probs) * np.log2(1 - probs + 1e-12)
        )
    _, ax = plt.subplots(figsize=(3.5, 3.5))
    im = ax.imshow(h, cmap=cmap, vmin=0, vmax=1)  # max entropy=1 when p=0.5
    for a in actions:
        r, c = divmod(a, n)
        ax.add_patch(
            Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                edgecolor="white",
                facecolor="none",
                linewidth=1.5,
            )
        )
    ax.set_title("Flip Entropy (bits)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="H")
    return ax
