"""
Evaluate trained square lattice RL models.

For each model, synthesizes N random linear functions and reports:
  - success rate (output == input up to linear equivalence)
  - average CNOT count
  - average CNOT depth (2-qubit gate depth)

Results are saved to a CSV file in the results/ directory.

Usage:
    # Evaluate local models (default paths)
    python evaluate_square_lattice.py

    # Evaluate local models at custom paths
    python evaluate_square_lattice.py --configs-dir ./my_configs --models-dir ./my_models

    # Evaluate models from HuggingFace
    python evaluate_square_lattice.py --repo-id DuruTo/ai-transpiler_linear-functions

    # Evaluate specific qubit sizes
    python evaluate_square_lattice.py --sizes 5 6

    # Evaluate specific topology letters
    python evaluate_square_lattice.py --topologies S B

    # Evaluate specific models
    python evaluate_square_lattice.py --names linear_function_5qS linear_function_6qB1

    # Control number of random samples per model
    python evaluate_square_lattice.py --num-samples 50
"""

import argparse
import csv
import os
import re
import time
from datetime import datetime

import numpy as np


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
from loguru import logger
from qiskit_gym.rl import RLSynthesis
from qiskit.circuit.library import LinearFunction
from qiskit.synthesis.linear.linear_matrix_utils import random_invertible_binary_matrix
from qiskit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


DEFAULT_CONFIGS_PATH = "./square_lattice_configs"
DEFAULT_MODELS_PATH = "./square_lattice_models"
RESULTS_PATH = "./results"


def parse_name(name: str) -> tuple[int, str]:
    m = re.match(r"linear_function_(\d+)q([A-Z]+)", name)
    if not m:
        raise ValueError(f"Cannot parse config name: {name}")
    return int(m.group(1)), m.group(2)


def filter_models(
    all_names: list[str],
    sizes: list[int] | None,
    topologies: list[str] | None,
    names: list[str] | None,
) -> list[str]:
    if names:
        return [n for n in all_names if n in names]
    result = all_names
    if sizes:
        result = [n for n in result if parse_name(n)[0] in sizes]
    if topologies:
        result = [n for n in result if parse_name(n)[1] in topologies]
    return result


def count_cnots(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.name == "cx")


def evaluate_model(
    name: str,
    configs_dir: str,
    models_dir: str,
    num_samples: int,
    num_searches: int,
) -> dict:
    config_path = os.path.join(configs_dir, f"{name}.json")
    safetensors_path = os.path.join(models_dir, f"{name}.safetensors")

    if not os.path.exists(safetensors_path):
        logger.warning(f"No safetensors found for {name}, skipping.")
        return None

    rls = RLSynthesis.from_config_json(config_path, safetensors_path)
    n = rls.env.config["num_qubits"]

    successes = 0
    cnot_counts = []
    depths = []

    for seed in range(num_samples):
        matrix = random_invertible_binary_matrix(n, seed=seed)
        input_qc = LinearFunction(matrix)
        qc = rls.synth(input_qc, num_searches=num_searches, num_mcts_searches=0, deterministic=False)

        try:
            correct = LinearFunction(qc) == LinearFunction(input_qc)
        except CircuitError:
            correct = False
        if correct:
            successes += 1
            cnot_counts.append(count_cnots(qc))
            depths.append(qc.depth(lambda x: x.operation.num_qubits == 2)) #2 qubit gate depth

    success_rate = successes / num_samples
    return {
        "name": name,
        "num_qubits": n,
        "success_rate": success_rate,
        "avg_cnots": np.mean(cnot_counts) if cnot_counts else float("nan"),
        "avg_cnot_depth": np.mean(depths) if depths else float("nan"),
        "num_samples": num_samples,
    }


def save_results_csv(results: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "num_qubits", "success_rate", "avg_cnots", "avg_cnot_depth", "num_samples", "timestamp", "duration_seconds"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate square lattice RL models.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--repo-id", type=str, help="HuggingFace repo ID to pull models from")
    source.add_argument("--configs-dir", type=str, default=DEFAULT_CONFIGS_PATH,
                        help=f"Path to configs directory (default: {DEFAULT_CONFIGS_PATH})")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_PATH,
                        help=f"Path to models directory (default: {DEFAULT_MODELS_PATH}). Ignored when --repo-id is set.")
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--topologies", nargs="+", type=str)
    parser.add_argument("--names", nargs="+", type=str)
    parser.add_argument("--num-samples", type=int, default=100, help="Random samples per model (default: 100)")
    parser.add_argument("--num-searches", type=int, default=100, help="Searches per synthesis call (default: 100)")
    args = parser.parse_args()

    if args.repo_id:
        from twisterl.utils import pull_hub_algorithm
        logger.info(f"Pulling models from {args.repo_id}...")
        snapshot_dir = pull_hub_algorithm(repo_id=args.repo_id, model_path="./models", revision="main", validate=False)
        if not snapshot_dir:
            logger.error("Failed to pull models from HuggingFace.")
            return
        configs_dir = snapshot_dir
        models_dir = snapshot_dir
    else:
        configs_dir = args.configs_dir
        models_dir = args.models_dir

    all_names = sorted(
        f[:-5] for f in os.listdir(configs_dir)
        if f.endswith(".json") and f.startswith("linear_function_")
        and os.path.exists(os.path.join(models_dir, f[:-5] + ".safetensors"))
    )

    selected = filter_models(
        all_names,
        sizes=args.sizes,
        topologies=[t.upper() for t in args.topologies] if args.topologies else None,
        names=args.names,
    )

    if not selected:
        logger.warning("No models matched the given filters (or no safetensors found).")
        return

    logger.info(f"Evaluating {len(selected)} model(s)...")
    start_time = time.time()
    results = []
    for name in selected:
        logger.info(f"Evaluating {name}...")
        result = evaluate_model(name, configs_dir, models_dir, args.num_samples, args.num_searches)
        if result:
            results.append(result)
            logger.info(
                f"  {name}: success={result['success_rate']:.1%} "
                f"avg_cnots={result['avg_cnots']:.1f} avg_cnot_depth={result['avg_cnot_depth']:.1f}"
            )

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'n':>3} {'Success':>8} {'Avg CX':>8} {'Avg CX Depth':>13}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<35} {r['num_qubits']:>3} "
            f"{r['success_rate']:>7.1%} {r['avg_cnots']:>8.1f} {r['avg_cnot_depth']:>13.1f}"
        )
    print("=" * 70)
    overall_success = np.mean([r["success_rate"] for r in results])
    print(f"{'Overall success rate:':<35} {overall_success:>7.1%}")

    duration = time.time() - start_time
    print(f"{'Total time:':<35} {duration / 60:.1f} min")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in results:
        r["timestamp"] = timestamp
        r["duration_seconds"] = round(duration, 1)
    csv_path = _next_path(RESULTS_PATH, "eval", "csv")
    save_results_csv(results, csv_path)


if __name__ == "__main__":
    main()
