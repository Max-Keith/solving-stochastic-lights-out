import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from lightsout.algebra import build_A, gf2_min_weight_solution
from lightsout.board import BoardState
from lightsout.evaluation.metrics import (
    presses_used,
    rescue_break,
    success_within_budget,
)
from lightsout.stochastic.kernel import (
    make_deterministic_kernel,
    make_exp_kernel,
)
from lightsout.stochastic.simulator import Simulator
from lightsout.strategies import (
    ChasingLights,
    ExpectedLightsGreedy,
    LinearAlgebraMinWeight,
    LinearAlgebraReplanning,
    ProbAtMostKGreedy,
    QLearning3x3,
    RandomClick,
    RiskAverseGreedy,
    StandardRollout,
    TwoStepExpectedGreedy,
)

# Limit threads per worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
mp.freeze_support()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_strategy(
    name: str, rng: np.random.Generator | None = None, params=None
):
    name = name.lower()
    if name == "random_click":
        return RandomClick(rng=rng)
    if name == "linear_algebra_minweight":
        return LinearAlgebraMinWeight()
    if name == "chasing_lights":
        return ChasingLights()
    if name == "greedy":
        return ExpectedLightsGreedy(
            rng=rng, tie_break=(params or {}).get("tie_break", "first")
        )
    if name == "risk_averse_greedy":
        return RiskAverseGreedy(
            rng=rng, tie_break=(params or {}).get("tie_break", "first")
        )
    if name == "two_step_greedy":
        return TwoStepExpectedGreedy(
            rng=rng, tie_break=(params or {}).get("tie_break", "first")
        )
    if name == "prob_greedy":
        return ProbAtMostKGreedy(
            max_on=(params or {}).get("max_on", 0),
            rng=rng,
            tie_break=(params or {}).get("tie_break", "first"),
        )
    if name in ("standard_rollout", "rollout"):
        return StandardRollout(rng=rng)
    if name == "linear_algebra_replanning":
        return LinearAlgebraReplanning(
            tie_break=(params or {}).get("tie_break", "first"),
        )
    if name == "q_learning_3x3":
        return QLearning3x3(
            q_dir=(params or {}).get("q_dir", "q_tables"),
            rng=rng,
            epsilon=(params or {}).get("epsilon", 0.0),
        )
    raise ValueError(f"Unknown strategy: {name}")


def parse_strategies(cfg_strats):
    """Parse strategy configs from YAML."""
    parsed = []
    for item in cfg_strats:
        if isinstance(item, str):
            parsed.append({"name": item, "params": {}})
        elif isinstance(item, dict) and "name" in item:
            d = dict(item)  # shallow copy
            d.setdefault("params", {})
            parsed.append({"name": d["name"], "params": d["params"]})
        else:
            raise ValueError(f"Invalid strategy spec: {item}")
    return parsed


def sample_initial_states(
    n: int, n_samples: int, rng: np.random.Generator
) -> list[BoardState]:
    """Sample random initial board states."""
    states = []
    max_lights = n * n

    for _ in range(n_samples):
        # Uniformly sample how many lights should be on (1 to n²)
        num_on = rng.integers(1, max_lights + 1)

        # Create board with exactly num_on lights
        flat = np.zeros(max_lights, dtype=bool)
        flat[:num_on] = True
        rng.shuffle(flat)

        s = BoardState(n, flat.reshape(n, n))
        states.append(s)

    return states


def _task_seed(base_seed: int, *coords: int) -> int:
    """Generate deterministic seed for each task."""
    ss = np.random.SeedSequence([int(base_seed)] + [int(c) for c in coords])

    return int(
        ss.generate_state(1, dtype=np.uint64)[0] & np.uint64((1 << 63) - 1)
    )


def make_batches(
    state_grids, kernels, alphas, lambdas, strat_specs, batch_size
):
    """Create job batches for parallel processing."""
    n_states = len(state_grids)
    ranges = [
        (i, min(i + batch_size, n_states))
        for i in range(0, n_states, batch_size)
    ]
    for spec in strat_specs:
        # Stochastic kernel jobs
        if "exp" in kernels:
            for a in alphas:
                for lmb in lambdas:
                    for lo, hi in ranges:
                        yield {
                            "kernel": "exp",
                            "alpha": float(a),
                            "lam": float(lmb),
                            "strategy_spec": spec,
                            "idx_lo": lo,
                            "idx_hi": hi,
                            "grids": state_grids[lo:hi],
                        }
        if "deterministic" in kernels:
            for lo, hi in ranges:
                yield {
                    "kernel": "deterministic",
                    "alpha": None,
                    "lam": None,
                    "strategy_spec": spec,
                    "idx_lo": lo,
                    "idx_hi": hi,
                    "grids": state_grids[lo:hi],
                }


def _run_batch(job):
    """Run one batch of simulations."""
    n = job["n"]
    max_steps = job["budget_T"]
    base_seed = job["base_seed"]
    grids = job["grids"]

    # Build kernel
    if job["kernel"] == "exp":
        P = make_exp_kernel(n, job["alpha"], job["lam"])
    else:
        P = make_deterministic_kernel(n)

    A = build_A(n)
    rows = []
    strat_name = job["strategy_spec"]["name"]
    strat_params = job["strategy_spec"]["params"] or {}

    # Per-batch seeds
    batch_coords = (
        1 if job["kernel"] == "deterministic" else 0,
        int((job["alpha"] or 0) * 1e6),
        int((job["lam"] or 0) * 1e6),
        job["idx_lo"],
        job["idx_hi"],
        hash(strat_name) & 0xFFFF,
    )
    sim_rng = np.random.default_rng(_task_seed(base_seed, *batch_coords, 1))
    simulator = Simulator(P, rng=sim_rng)

    for offset, grid in enumerate(grids):
        board_id = job["idx_lo"] + offset
        init = BoardState(n, grid)

        # Build strategy with its own deterministic seed
        run_rng = np.random.default_rng(
            _task_seed(base_seed, *batch_coords, board_id, 2)
        )
        strat = make_strategy(strat_name, rng=run_rng, params=strat_params)
        params_for_reset = {
            "P": P,
            "rng": run_rng,
            "alpha": job["alpha"],
            "lambda": job["lam"],
            **strat_params,
        }
        strat.reset(n, params=params_for_reset)

        # Check if deterministically solvable
        target = init.to_flat().astype(np.uint8)
        _, is_solvable = gf2_min_weight_solution(A, target)

        # Run simulation
        start_time = time.perf_counter()
        sim_ret = simulator.run(init, strat, T=max_steps)
        end_time = time.perf_counter()
        time_ms = (end_time - start_time) * 1000

        counts = sim_ret[2]
        solved = int(success_within_budget(counts, max_steps))
        num_presses = int(presses_used(counts))
        time_per_action = time_ms / num_presses if num_presses > 0 else 0.0
        rescued, broken = rescue_break(int(is_solvable), solved)

        rows.append(
            {
                "kernel": job["kernel"],
                "n": n,
                "alpha": (
                    ""
                    if job["kernel"] == "deterministic"
                    else float(job["alpha"])
                ),
                "lambda": (
                    ""
                    if job["kernel"] == "deterministic"
                    else float(job["lam"])
                ),
                "strategy": strat_name,
                "seed": base_seed,
                "board_id": board_id,
                "initial_on": init.count_on(),
                "det_solvable": int(is_solvable),
                "solved": solved,
                "presses_used": num_presses,
                "time_ms": time_ms,
                "time_per_action_ms": time_per_action,
                "rescued": int(rescued),
                "broken": int(broken),
            }
        )
    return rows


def run_pool(jobs, writer, workers, max_inflight=None, total_jobs=None):
    """Run jobs in parallel and write results as they complete."""
    ctx = mp.get_context("spawn")
    if max_inflight is None:
        max_inflight = workers * 3

    inflight = set()
    submitted = 0
    done = 0
    total_rows = 0
    start_time = time.time()

    if total_jobs is None and hasattr(jobs, "__len__"):
        try:
            total_jobs = len(jobs)
        except Exception:
            total_jobs = None

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        jobs_iter = iter(jobs)
        while len(inflight) < max_inflight:
            try:
                j = next(jobs_iter)
            except StopIteration:
                break
            inflight.add(ex.submit(_run_batch, j))
            submitted += 1

        while inflight:
            for fut in as_completed(inflight, timeout=None):
                inflight.remove(fut)
                try:
                    rows = fut.result()
                except Exception:
                    import traceback

                    print("\n[ERROR] Worker failed:")
                    traceback.print_exc()
                    raise
                writer.writerows(rows)
                done += 1
                total_rows += len(rows)

                # Progress reporting
                if done % 1 == 0 or done == total_jobs:
                    elapsed = time.time() - start_time
                    if total_jobs:
                        pct = done / total_jobs
                        if done > 0:
                            eta_seconds = (elapsed / done) * (total_jobs - done)
                            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                        else:
                            eta_str = "N/A"
                        progress_line = (
                            f"\r[progress] {done}/{total_jobs} batches ({pct:>6.1%}) | "
                            f"{total_rows:>7,} boards | "
                            f"elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s | "
                            f"ETA: {eta_str}"
                        )
                        print(progress_line, end="", flush=True)
                        if done == total_jobs:
                            print()  # Final newline
                    else:
                        # Fallback: report against submitted if total is unknown
                        pct = done / max(submitted, 1)
                        progress_line = (
                            f"\r[progress] {done}/{submitted} batches (~{pct:.0%}) | "
                            f"{total_rows:>7,} boards | "
                            f"elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s"
                        )
                        print(progress_line, end="", flush=True)
                # Submit next job to keep inflight bounded
                try:
                    j = next(jobs_iter)
                    inflight.add(ex.submit(_run_batch, j))
                    submitted += 1
                except StopIteration:
                    pass
                break  # re-enter as_completed with updated set


def main():
    n_cpus = os.cpu_count() or 1
    default_workers = max(n_cpus - 1, 1)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "configs" / "sweep_5x5.yaml"),
    )
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument(
        "--workers", type=int, default=default_workers, help="Number of workers"
    )
    ap.add_argument(
        "--batch-size", type=int, default=1000, help="Boards per batch"
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["experiment"]

    n = int(cfg["board"]["n"])
    kernels = list(cfg["kernels"])
    alphas = list(cfg["noise"]["alpha"])
    lambdas = list(cfg["noise"]["lambda"])
    max_steps = int(cfg["budget_T"])
    n_samples = int(cfg["initial_states"]["n_samples"])
    base_seed = int(cfg["initial_states"].get("seed", 0))
    out_dir = Path(cfg.get("output_dir", "results/runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out or str(out_dir / "practice.csv")

    strat_specs = parse_strategies(cfg["strategies"])

    # Sample initial states
    rng = np.random.default_rng(base_seed)
    states = sample_initial_states(n, n_samples, rng)
    state_grids = [
        s.to_flat().reshape(n, n).astype(bool, copy=True) for s in states
    ]

    # Compute total number of batch jobs for accurate progress
    n_states = len(state_grids)
    num_ranges = (n_states + args.batch_size - 1) // args.batch_size
    num_strats = len(strat_specs)
    exp_per_strat = (len(alphas) * len(lambdas)) if ("exp" in kernels) else 0
    det_per_strat = 1 if ("deterministic" in kernels) else 0
    total_jobs = num_ranges * num_strats * (exp_per_strat + det_per_strat)

    def job_stream():
        for j in make_batches(
            state_grids, kernels, alphas, lambdas, strat_specs, args.batch_size
        ):
            j.update({"n": n, "budget_T": max_steps, "base_seed": base_seed})
            yield j

    fieldnames = [
        "n",
        "kernel",
        "alpha",
        "lambda",
        "strategy",
        "seed",
        "board_id",
        "initial_on",
        "det_solvable",
        "solved",
        "presses_used",
        "time_ms",
        "time_per_action_ms",
        "rescued",
        "broken",
    ]

    print(
        f"\nStarting {total_jobs:,} batches ({n_states:,} boards) with {args.workers} workers...\n"
    )

    start_time = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        run_pool(
            job_stream(),
            writer,
            workers=args.workers,
            max_inflight=args.workers * 3,
            total_jobs=total_jobs,
        )

    elapsed = time.time() - start_time
    print(f"\nDone in {int(elapsed/60)}m {int(elapsed%60)}s")
    print(f"Output: {out_csv}\n")


if __name__ == "__main__":
    main()
