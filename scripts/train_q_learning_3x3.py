import numpy as np

from lightsout.rl.utils_3x3 import array_to_index_3x3
from lightsout.stochastic.kernel import make_exp_kernel


def step_env_3x3(
    s_flat: np.ndarray, a: int, P: np.ndarray, rng
) -> tuple[np.ndarray, float, bool]:
    """Take one step in the 3x3 Lights Out environment."""
    s = s_flat.copy().astype(np.int8)

    # Flip cells according to kernel probabilities
    p_flip = P[a]
    flips = rng.random(9) < p_flip
    s = np.logical_xor(s, flips).astype(np.int8)

    done = bool(s.sum() == 0)

    # Give small negative reward each step to encourage faster solving
    reward = 0.0 if done else -1.0

    return s, reward, done


def train_q_learning_3x3(
    alpha_noise: float,
    lambda_noise: float,
    num_episodes: int = 20_000,
    max_steps_per_episode: int = 30,
    gamma: float = 0.99,
    lr_start: float = 0.3,
    lr_end: float = 0.01,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
):
    """Train a Q-table for 3x3 Lights Out with given noise parameters."""
    rng = np.random.default_rng(seed)

    n = 3
    N = n * n

    # Build stochastic kernel for this noise level
    P = make_exp_kernel(n, alpha_noise, lambda_noise)

    # Initialize Q-table (512 states x 9 actions)
    Q = np.zeros((2**N, N), dtype=np.float32)

    # Linear decay schedules
    def epsilon_schedule(ep):
        t = min(1.0, ep / (num_episodes * 0.8))
        return epsilon_start * (1 - t) + epsilon_end * t

    def lr_schedule(ep):
        t = min(1.0, ep / (num_episodes * 0.8))
        return lr_start * (1 - t) + lr_end * t

    for ep in range(num_episodes):
        # Random initial board
        init_bits = (rng.random(N) < 0.5).astype(np.int8)
        s = init_bits
        s_idx = array_to_index_3x3(s)

        eps = epsilon_schedule(ep)
        lr = lr_schedule(ep)

        for t in range(max_steps_per_episode):
            # Epsilon-greedy action selection
            if rng.random() < eps:
                a = int(rng.integers(0, N))
            else:
                a = int(np.argmax(Q[s_idx]))

            # Take environment step
            s_next, reward, done = step_env_3x3(s, a, P, rng)
            s_next_idx = array_to_index_3x3(s_next)

            # Q-learning update
            best_next = 0.0 if done else np.max(Q[s_next_idx])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[s_idx, a]

            Q[s_idx, a] += lr * td_error

            s, s_idx = s_next, s_next_idx

            if done:
                break

        if verbose and (ep + 1) % 5000 == 0:
            print(f"[train] episode {ep+1}/{num_episodes}")

    return Q


if __name__ == "__main__":
    alpha = 0.05
    lam = 0.5

    Q = train_q_learning_3x3(alpha, lam, verbose=True)
    np.save(f"q_table_3x3_alpha{alpha}_lambda{lam}.npy", Q)
    print("Saved Q-table.")
