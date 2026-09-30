# squarebench

Train and benchmark RL-based linear-function synthesis on **square-lattice** topologies, and compare against Qiskit and the Qiskit transpiler service on the Benchpress suite.

## Contents

- [Setup](#setup)
- [Running benchmarks (Benchpress)](#running-benchmarks-benchpress)
- [Project pipeline](#project-pipeline)
  - [1. Pretraining — enumerate topologies](#1-pretraining--enumerate-topologies)
  - [2. Training — train RL models](#2-training--train-rl-models)
  - [3. Evaluation — measure model quality](#3-evaluation--measure-model-quality)
  - [4. Plotting — visualise results](#4-plotting--visualise-results)
- [Repository layout](#repository-layout)

## Setup

Requires **Python 3.10**.

```bash
# create + activate venv
python3.10 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install benchpress
pip install git+https://github.com/Qiskit/benchpress

# install qiskit_gym 0.4.x (older versions fail)
pip install git+https://github.com/AI4quantum/qiskit-gym.git@0.4.0
```

## Running benchmarks (Benchpress)

Benchpress runs as a pytest suite. Use `--benchmark-save=<tag>` to write results to `.benchmarks/` for later comparison and plotting.

```bash
# Qiskit baseline on the qiskit_gym suite
python -m pytest --timeout-skip-list=600 \
  --benchmark-save=qiskit \
  benchpress/qiskit_gym

# Qiskit transpiler service (AI) on the Summit test
python -m pytest --timeout-skip-list=600 \
  --benchmark-save=ai-test-summit \
  benchpress/qiskit_transpiler_service_gym/test_summit_qts.py
```

Other entry points:

| Path | Suite |
|---|---|
| [benchpress/qiskit_gym/test_summit.py](benchpress/qiskit_gym/test_summit.py) | Qiskit on Summit input circuits |
| [benchpress/qiskit_gym/test_summit_sqr.py](benchpress/qiskit_gym/test_summit_sqr.py) | Qiskit + square-lattice RL on Summit |
| [benchpress/qiskit_transpiler_service_gym/test_summit_qts.py](benchpress/qiskit_transpiler_service_gym/test_summit_qts.py) | Qiskit transpiler service on Summit |
| [benchpress/test_qasm_miami.py](benchpress/test_qasm_miami.py) | QASMBench on the Miami backend |

## Project pipeline

### 1. Pretraining — enumerate topologies

Generate all unique connected subgraphs (2–8 nodes) of the 5×5 square lattice:
```bash
python pretraining/generate_square_lattice_subgraphs.py
# outputs: unique_subgraphs.pkl
```

Generate the training config JSON files for each topology:
```bash
python pretraining/generate_square_lattice_configs.py
# outputs: square_lattice_configs/linear_function_<N>q<T>.json
```

Optionally check which topologies are already covered by existing heavy-hex models:
```bash
python pretraining/check_heavy_hex_coverage.py
```

### 2. Training — train RL models

```bash
# Train all configs
python training/train_square_lattice.py

# Train a subset (e.g. 7- and 8-qubit S/B topologies)
python training/train_square_lattice.py --sizes 7 8 --topologies S B
# outputs: square_lattice_models/
```

### 3. Evaluation — measure model quality

Evaluate success rate and CNOT count for each trained model:
```bash
python evaluation/evaluate_square_lattice.py
# outputs: results/evaluate_<timestamp>.csv
```

Benchmark RL synthesis on random linear functions:
```bash
python evaluation/benchmark_linear_functions.py
```

Merge benchmark runs / check isomorphism between topologies:
```bash
python evaluation/merge_benchmark.py
python evaluation/check_isomorphism.py
```

### 4. Plotting — visualise results

| Script | Input | What it shows |
|---|---|---|
| [plotting/plot_model_correctness.py](plotting/plot_model_correctness.py) | `results/evaluate_*.csv` | Per-model success rate after training |
| [plotting/plot_training_progress.py](plotting/plot_training_progress.py) | TensorBoard runs in `runs/` | Training curves by qubit size |
| [plotting/plot_training_duration.py](plotting/plot_training_duration.py) | `runs/` | Wall-clock training time per topology |
| [plotting/plot_variance.py](plotting/plot_variance.py) | benchmark CSVs | Std-dev of 2Q gate count / depth across methods |
| [plotting/plot_benchpress_summit.py](plotting/plot_benchpress_summit.py) | Benchpress JSON output | 2Q gate count & depth vs Qiskit on the Summit suite |
| [plotting/plot_random_lf_circuits.py](plotting/plot_random_lf_circuits.py) | benchmark JSONs | Random linear-function circuit comparisons |
| [plotting/plot_topology.py](plotting/plot_topology.py) | `square_lattice_configs/*.json` | Coupling-map diagram for a single topology |

Example:
```bash
python plotting/plot_model_correctness.py results/evaluate_latest.csv
python plotting/plot_topology.py square_lattice_configs/linear_function_8qE.json -o topology_8qE.png
```

## Repository layout

```
benchpress/             # Benchpress fork — pytest-based benchmark suites
  qiskit_gym/             # Qiskit + square-lattice RL workouts
  qiskit_transpiler_service_gym/   # Qiskit transpiler service (AI) workouts
  qasm/                   # QASM circuit corpora (QASMBench, QV, QFT, …)
pretraining/            # Topology enumeration + config generation
training/               # RL training entry points
evaluation/             # Success-rate / gate-count evaluation
plotting/               # Plot scripts (consume CSV / JSON outputs)
square_lattice_configs/ # Generated per-topology training configs
square_lattice_models/  # Trained model checkpoints
results/                # CSVs, JSONs, and rendered plots
runs/                   # TensorBoard run logs
```
