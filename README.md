It has to be python version 3.10
```bash
python3.10 -m venv .venv
```
install dependencies:
```bash
pip install -r req
```
install benchpress:
```bash
pip install git+https://github.com/Qiskit/benchpress
```


install qiskit_gym 0.4.x inside .venv/python3.10/site-packages (older versions give error):
````bash
pip install git+https://github.com/AI4quantum/qiskit-gym.git@0.4.0
```
commands: 
python -m pytest --timeout-skip-list=600 --benchmark-save=qiskit  benchpress/qiskit_gym
python -m pytest --timeout-skip-list=600 --benchmark-save=ai-test-summit  benchpress/qiskit_transpiler_service_gym/device_transpile/test_summit.py


## Project Pipeline

### 1. Pretraining — enumerate topologies

Generate all unique connected subgraphs (2–8 nodes) of the 5×5 square lattice and save them:
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

Compare RL synthesis against Qiskit transpilation on the same circuits:
```bash
python evaluation/compare_qiskit.py
# outputs: results/compare_<timestamp>.csv
```

### 4. Plotting — visualise results

| Script | Input | What it shows |
|---|---|---|
| `plotting/plot_model_correctness.py` | `results/evaluate_*.csv` | Per-model success rate after training |
| `plotting/plot_training_progress.py` | TensorBoard runs in `runs/` | Training curves by qubit size |
| `plotting/plot_training_duration.py` | `runs/` | Wall-clock training time per topology |
| `plotting/plot_variance.py` | `results/compare_*.csv` | Std-dev of 2Q gate count / depth across methods |
| `plotting/plot_benchpress_summit.py` | Benchpress JSON output | 2Q gate count & depth vs Qiskit on the Summit suite |
| `plotting/plot_topology.py` | `square_lattice_configs/*.json` | Coupling-map diagram for a single topology |

Example:
```bash
python plotting/plot_model_correctness.py results/evaluate_latest.csv
python plotting/plot_topology.py square_lattice_configs/linear_function_8qE.json -o topology_8qE.png
```
