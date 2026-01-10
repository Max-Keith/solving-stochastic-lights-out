import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from train_q_learning_3x3 import train_q_learning_3x3

# Limit threads per worker to avoid over-subscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def train_and_save(job):
    """Train one Q-table and save it to disk."""
    alpha = job["alpha"]
    lam = job["lambda"]
    params = job["params"]

    start = time.time()
    Q = train_q_learning_3x3(alpha_noise=alpha, lambda_noise=lam, **params)
    duration = time.time() - start

    fname = f"results/q_tables/q_3x3_alpha{alpha}_lambda{lam}.npy"
    np.save(fname, Q)

    return {"alpha": alpha, "lambda": lam, "time": duration}


def main():
    # Noise parameter combinations
    alphas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    lambdas = [0.0001, 0.5, 1.0, 2.0, 5.0]

    params = {
        "num_episodes": 100_000,
        "max_steps_per_episode": 50,
        "gamma": 0.95,
        "lr_start": 0.3,
        "lr_end": 0.01,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "seed": 0,
    }

    Path("results/q_tables").mkdir(parents=True, exist_ok=True)

    # Create a job for each (alpha, lambda) combination
    jobs = []
    for a in alphas:
        for lam in lambdas:
            jobs.append({"alpha": a, "lambda": lam, "params": params})

    n_jobs = len(jobs)
    n_workers = max((os.cpu_count() or 1) - 1, 1)

    print(
        f"\nTraining {n_jobs} Q-tables in parallel with {n_workers} workers..."
    )
    print(f"Alphas: {alphas}")
    print(f"Lambdas: {lambdas}\n")

    # Run training jobs in parallel
    ctx = mp.get_context("spawn")
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(train_and_save, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                completed += 1
                elapsed = time.time() - start_time
                # Simple ETA based on average time per job
                eta = (elapsed / completed) * (n_jobs - completed)

                print(
                    f"[{completed}/{n_jobs}] "
                    f"alpha={result['alpha']:.2f}, lambda={result['lambda']:.4f} "
                    f"({result['time']:.0f}s, ETA: {int(eta/60)}m)"
                )
            except Exception as e:
                completed += 1
                print(
                    f"[{completed}/{n_jobs}] FAILED "
                    f"alpha={job['alpha']}, lambda={job['lambda']}: {e}"
                )

    total_time = time.time() - start_time

    print(
        f"\nDone! Trained {n_jobs} Q-tables in {int(total_time/60)}m {int(total_time%60)}s"
    )
    print(f"Saved to results/q_tables/\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
