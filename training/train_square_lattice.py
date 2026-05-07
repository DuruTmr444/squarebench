"""
Train RL models for square lattice topologies.

Usage:
    # Train all configs
    python train_square_lattice.py

    # Train specific qubit sizes
    python train_square_lattice.py --sizes 5 6

    # Train specific topology letters
    python train_square_lattice.py --topologies S B

    # Train specific files
    python train_square_lattice.py --names linear_function_5qS linear_function_6qB1

    # Combine filters (AND logic)
    python train_square_lattice.py --sizes 7 8 --topologies S
"""

import argparse
import json
import os
import re

from loguru import logger
from safetensors.torch import save_file
from qiskit_gym.rl import RLSynthesis, PPOConfig, BasicPolicyConfig
from torch.utils.tensorboard import SummaryWriter


CONFIG_DIR = "square_lattice_configs"
OUTPUT_DIR = "square_lattice_models"
RUNS_DIR = "runs"


def load_rls_from_config(config_path: str) -> tuple[RLSynthesis, int]:
    """Load RLSynthesis from a config JSON, return (rls, diff_max)."""
    rls = RLSynthesis.from_config_json(config_path)
    with open(config_path) as f:
        config = json.load(f)
    diff_max = config["algorithm"]["learning"]["diff_max"]
    return rls, diff_max


def train(
    rls: RLSynthesis,
    tb_path: str,
    diff_max: int,
    chunk_size: int = 50,
    patience: int = 10,
    min_improvement: float = 0.005,
) -> float:
    """
    Train until max difficulty is reached and reward plateaus.

    Calls rls.algorithm.learn() directly (not rls.learn()) to avoid
    resetting difficulty between chunks.

    Args:
        chunk_size: training iterations per chunk before checking early stopping
        patience: chunks with no reward improvement (at max difficulty) before stopping
        min_improvement: minimum reward delta to count as improvement

    Returns:
        best reward achieved
    """
    rls.algorithm.run_path = tb_path
    rls.algorithm.tb_writer = SummaryWriter(tb_path)
    os.makedirs(tb_path, exist_ok=True)

    eval_cfg = rls.algorithm.config["evals"]["ppo_deterministic"]
    best_reward = -float("inf")
    chunks_without_improvement = 0
    iteration = 0

    while True:
        rls.algorithm.learn(chunk_size)
        iteration += chunk_size

        (success, reward), _ = rls.algorithm.evaluate(eval_cfg)
        at_max = rls.env.difficulty >= diff_max

        if reward > best_reward + min_improvement:
            best_reward = reward
            chunks_without_improvement = 0
        elif at_max:
            chunks_without_improvement += 1

        logger.info(
            f"iter={iteration} diff={rls.env.difficulty}/{diff_max} "
            f"success={success:.3f} reward={reward:.4f} "
            f"no_improve_chunks={chunks_without_improvement}/{patience}"
        )

        if at_max and chunks_without_improvement >= patience:
            logger.info("Early stopping: max difficulty reached and reward plateaued.")
            break

    return best_reward


def parse_name(name: str) -> tuple[int, str]:
    """Parse 'linear_function_NqT[index]' -> (N, T)."""
    m = re.match(r"linear_function_(\d+)q([A-Z]+)", name)
    if not m:
        raise ValueError(f"Cannot parse config name: {name}")
    return int(m.group(1)), m.group(2)


def filter_configs(
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


def main():
    parser = argparse.ArgumentParser(description="Train square lattice RL models.")
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="Qubit counts to train (e.g. 5 6 7)"
    )
    parser.add_argument(
        "--topologies", nargs="+", type=str, help="Topology letters to train (e.g. S B X)"
    )
    parser.add_argument(
        "--names", nargs="+", type=str, help="Specific config names (without .json)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50, help="Training iterations per chunk (default: 50)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Chunks without improvement before stopping (default: 10)"
    )
    parser.add_argument(
        "--min-improvement", type=float, default=0.005, help="Min reward delta to count as improvement (default: 0.005)"
    )
    args = parser.parse_args()

    all_configs = sorted(
        f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".json")
    )

    selected = filter_configs(
        all_configs,
        sizes=args.sizes,
        topologies=[t.upper() for t in args.topologies] if args.topologies else None,
        names=args.names,
    )

    if not selected:
        logger.warning("No configs matched the given filters.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Training {len(selected)} model(s): {selected}")

    for name in selected:
        safetensors_path = os.path.join(OUTPUT_DIR, f"{name}.safetensors")
        if os.path.exists(safetensors_path):
            logger.info(f"Skipping {name} (already trained).")
            continue

        config_path = os.path.join(CONFIG_DIR, f"{name}.json")
        safetensors_path = os.path.join(OUTPUT_DIR, f"{name}.safetensors")
        tb_path = os.path.join(RUNS_DIR, name)

        logger.info(f"\n{'='*60}\nStarting: {name}\n{'='*60}")

        rls, diff_max = load_rls_from_config(config_path)
        rls.env.difficulty = 1

        best_reward = train(
            rls,
            tb_path=tb_path,
            diff_max=diff_max,
            chunk_size=args.chunk_size,
            patience=args.patience,
            min_improvement=args.min_improvement,
        )

        save_file(rls.algorithm.policy.state_dict(), safetensors_path)
        logger.info(f"Saved: {safetensors_path} (best_reward={best_reward:.4f})")


if __name__ == "__main__":
    main()
