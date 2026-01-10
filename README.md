# Solving Stochastic Lights Out

Public companion repository for the bachelor thesis:

**_Solving Stochastic Lights Out: A Comparison of Greedy and Planning Strategies_**  
Max Steinke

This repo contains the core implementation, experiment configs, scripts, result files, and analysis notebooks used to compare greedy and planning-based strategies in **stochastic** Lights Out.

---

## Repository structure

- `src/lightsout/` — main Python package (board/algebra, stochastic simulator, strategies, evaluation)
- `scripts/` — training + experiment runners
- `experiments/configs/` — YAML configs for sweeps and evaluations
- `notebooks/` — analysis notebooks used for plots and statistics
- `results/runs/` — CSV outputs (including `results/runs/final/` used for thesis figures)
- `results/plots/` — final plots used in the thesis
- `results/q_tables/` — trained Q-tables (`.npy`) for 3x3 experiments

---

## Quick start

```bash
# (Recommended) create & activate a virtual env first
pip install -e .  # installs the 'lightsout' package from src/
```

## Running experiments

### Run a sweep / experiment from a config

Use the provided runner:

```bash
python -u scripts/run_parallel.py --config experiments/configs/sweep_5x5.yaml --out results/runs/my_run.csv
```

Available configs:

- `experiments/configs/sweep_5x5.yaml`
- `experiments/configs/budget.yaml`
- `experiments/configs/q_eval_3x3.yaml`

Outputs are written as CSV to `results/runs/`.

## Q-learning (3x3)

To train multiple Q-tables (saved to `results/q_tables/`):

```bash
python -u scripts/train_multiple_q_learning.py
```

The notebook `notebooks/04_q_learning_analysis.ipynb` analyzes the trained tables and comparisons.

## Reproducing the analysis and plots

The analysis notebooks are in `notebooks/`:

- `01_stochastic_check.ipynb`: analysis of stochastic behavior
- `02_prelim_test_analysis.ipynb`: preliminary test analysis and plots
- `03_statistical_analysis.ipynb`: statistical tests and final plots
- `04_q_learning_analysis.ipynb`: Q-learning results and comparisons

Final plots used for the thesis are stored in `results/plots/`.
CSV files used for the final comparisons are in `results/runs/final/`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
