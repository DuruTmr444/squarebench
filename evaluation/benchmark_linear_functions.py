"""
Benchmark: random linear functions (4–8 qubits) transpiled for FakeMiami or FakeMarrakesh.

Three methods compared:
  1. Qiskit        — generate_preset_pass_manager (opt level 3)
  2. Qiskit HF AI  — generate_ai_pass_manager with default Qiskit repo
  3. User HF AI    — generate_ai_pass_manager with DuruTo repo

Metrics: average 2-qubit gate count and 2-qubit gate depth.
Results are written to results/benchmark_<backend>_<timestamp>.csv.
Visualisation: run plotting/plot_random_lf_circuits.py on the generated CSV.

Usage:
    python benchmark_miami.py
    python benchmark_miami.py --num-samples 50 --opt-level 1
    python benchmark_miami.py --qubit-counts 4 5 6
    python benchmark_miami.py --backend marrakesh
"""

import argparse
import csv
import logging
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _next_path(directory, stem, ext):
    candidate = os.path.join(directory, f"{stem}.{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(directory, f"{stem}_{n}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1

sys.path.insert(0, ".")

SQR_REPO = "DuruTo/ai-transpiler_linear-functions"
QISKIT_REPO = "Qiskit/ai-transpiler_linear-functions"

RESULTS_PATH = "./results"


def count_2q(qc):
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    #return qc.count_ops().get("cz")


def depth_2q(qc):
    return qc.depth(lambda x: x.operation.num_qubits == 2)


def make_qiskit_pm(backend, opt_level):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    return generate_preset_pass_manager(optimization_level=opt_level, backend=backend)


def make_ai_pm(backend, repo_id, opt_level):
    os.environ["QISKIT_TRANSPILER_LINEAR_FUNCTION_REPO_ID"] = repo_id

    # Reset cached repositories so the new repo_id is picked up.
    from qiskit_ibm_transpiler.model_bootstrap import reset_model_repository
    reset_model_repository()

    from qiskit_ibm_transpiler import generate_ai_pass_manager
    return generate_ai_pass_manager(
        optimization_level=opt_level,
        ai_optimization_level=opt_level,
        backend=backend,
    )


def run_method(pm, circuit):
    try:
        result = pm.run(circuit)
        return count_2q(result), depth_2q(result)
    except Exception:
        return None, None
# ergebnis schaltung passt auf die topologie?
# SABRE auf die Schaltung laufen lassen, gucken ob sich was verändert, wenn ja, ist was falsch

def make_backend(backend_name=None):
    if backend_name is None:
        from benchpress.config import Configuration
        backend_name = Configuration.options.get("general", {}).get("backend_name", "miami")
        # Normalise: 'fake_miami' -> 'miami', 'fake_marrakesh' -> 'marrakesh', etc.
        backend_name = backend_name.removeprefix("fake_")

    if backend_name == "miami":
        from benchpress.qiskit_gym.utils.miami import FakeMiami
        return FakeMiami()
    elif backend_name == "marrakesh":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        return FakeMarrakesh()
    else:
        raise ValueError(f"Unknown backend: {backend_name!r}. Choose 'miami' or 'marrakesh'.")


def benchmark(qubit_counts, num_samples, opt_level, backend_name):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import LinearFunction
    from qiskit.synthesis.linear.linear_matrix_utils import random_invertible_binary_matrix

    backend = make_backend(backend_name)

    # Build pass managers once per method (model loading is expensive).
    print("Initialising pass managers...")
    pm_qiskit = make_qiskit_pm(backend, opt_level)
    pm_qiskit_hf = make_ai_pm(backend, QISKIT_REPO, opt_level)
    pm_duru = make_ai_pm(backend, SQR_REPO, opt_level)

    methods = [
        ("qiskit",    pm_qiskit),
        ("qiskit_hf", pm_qiskit_hf),
        ("duru",      pm_duru),
    ]

    rows = []
    for n in qubit_counts:
        print(f"\n--- {n} qubits ---")
        stats = {name: {"cx": [], "depth": []} for name, _ in methods}

        for seed in range(num_samples):
            matrix = random_invertible_binary_matrix(n, seed=seed)
            input_lf = LinearFunction(matrix)
            qc = QuantumCircuit(n)
            qc.append(input_lf, range(n))

            for name, pm in methods:
                cx, d = run_method(pm, qc)
                if cx is not None:
                    stats[name]["cx"].append(cx)
                    stats[name]["depth"].append(d)

        for name, _ in methods:
            cx_vals = stats[name]["cx"]
            d_vals = stats[name]["depth"]
            avg_cx    = np.mean(cx_vals) if cx_vals else float("nan")
            std_cx    = np.std(cx_vals)  if cx_vals else float("nan")
            avg_depth = np.mean(d_vals)  if d_vals  else float("nan")
            std_depth = np.std(d_vals)   if d_vals  else float("nan")
            n_ok = len(cx_vals)
            print(f"  {name:<12}  avg_2q={avg_cx:.1f}±{std_cx:.1f}  avg_depth={avg_depth:.1f}±{std_depth:.1f}  ({n_ok}/{num_samples} ok)")
            rows.append({
                "num_qubits": n,
                "method": name,
                "num_samples": num_samples,
                "n_success": n_ok,
                "avg_2q_gates": round(avg_cx, 3),
                "std_2q_gates": round(std_cx, 3),
                "avg_2q_depth": round(avg_depth, 3),
                "std_2q_depth": round(std_depth, 3),
            })

    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubit-counts", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--backend", default=None, choices=["miami", "marrakesh"],
                        help="Fake backend to use (default: read from benchpress default.conf, fallback: miami)")
    args = parser.parse_args()

    start = time.time()
    rows = benchmark(args.qubit_counts, args.num_samples, args.opt_level, args.backend)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed / 60:.1f} min")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    csv_path = _next_path(RESULTS_PATH, f"benchmark_{args.backend}", "csv")
    fieldnames = ["num_qubits", "method", "num_samples", "n_success", "avg_2q_gates", "std_2q_gates", "avg_2q_depth", "std_2q_depth"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {csv_path}")
    print(f"To visualise: python plotting/plot_random_lf_circuits.py --csv {csv_path}")


if __name__ == "__main__":
    main()
