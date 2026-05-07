"""
Plot training progress curves for a given qubit size from TensorBoard runs.

Reads TensorBoard event files from the runs/ directory and plots three panels:
  - Reward
  - Success Rate
  - Difficulty

All models of the selected qubit size are overlaid in a single figure,
enabling convergence comparison across coupling-map topologies.

The x-axis represents total linear functions seen (gradient steps × 1024 episodes/step).

Usage:
    python plotting/plot_training_progress.py --qubits 8
    python plotting/plot_training_progress.py --qubits 6 --runs-dir path/to/runs
    python plotting/plot_training_progress.py --qubits 7 --configs-dir square_lattice_configs
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS_PATH = "./results/plot_training"
RUNS_DIR = "./runs"
EPISODES_PER_STEP = 1024

TAGS = ["Benchmark/reward", "Benchmark/success", "Benchmark/difficulty"]
LABELS = {
    "Benchmark/reward":     "Reward",
    "Benchmark/success":    "Success Rate",
    "Benchmark/difficulty": "Difficulty",
}


def load_run(run_path, tags):
    ea = EventAccumulator(run_path, size_guidance={"tensors": 0})
    ea.Reload()
    result = {}
    for tag in tags:
        try:
            events = ea.Tensors(tag)
        except KeyError:
            continue
        steps = [e.step for e in events]
        vals  = [e.tensor_proto.float_val[0] for e in events]
        global_steps = []
        offset = 0
        for i, s in enumerate(steps):
            if i > 0 and s < steps[i - 1]:
                offset = global_steps[-1] + 1
            global_steps.append(offset + s)
        result[tag] = (np.array(global_steps) * EPISODES_PER_STEP, np.array(vals))
    return result


def fmt_millions(x, _):
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}k"
    return str(int(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True,
                        help="Qubit size to plot (e.g. 8)")
    parser.add_argument("--runs-dir", type=str, default=RUNS_DIR,
                        help=f"Path to TensorBoard runs directory (default: {RUNS_DIR})")
    parser.add_argument("--configs-dir", type=str, default=None,
                        help="Optional config directory; only runs with matching <name>.json are plotted")
    args = parser.parse_args()

    prefix = f"linear_function_{args.qubits}q"
    model_names = sorted(
        d for d in os.listdir(args.runs_dir)
        if d.startswith(prefix) and os.path.isdir(os.path.join(args.runs_dir, d))
    )

    if args.configs_dir:
        config_names = {
            os.path.splitext(name)[0]
            for name in os.listdir(args.configs_dir)
            if name.startswith(prefix) and name.endswith(".json")
        }
        model_names = [name for name in model_names if name in config_names]

    if not model_names:
        where = f"{args.runs_dir}/"
        if args.configs_dir:
            where += f" with matching configs in {args.configs_dir}/"
        print(f"No runs found for {args.qubits}-qubit models (looked for '{prefix}*' in {where})")
        return

    print(f"Plotting {len(model_names)} run(s): {', '.join(model_names)}")

    cmap = plt.cm.get_cmap("tab10", max(len(model_names), 1))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Training Progress — {args.qubits}q Models", fontsize=14)

    for ax, tag in zip(axes, TAGS):
        for i, name in enumerate(model_names):
            data = load_run(os.path.join(args.runs_dir, name), [tag])
            if tag not in data:
                continue
            x, y = data[tag]
            ax.plot(x, y, color=cmap(i), linewidth=1.2, alpha=0.85,
                    label=name.replace("linear_function_", ""))
        ax.set_ylabel(LABELS[tag])
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Linear Functions Seen")
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(fmt_millions))
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = os.path.join(RESULTS_PATH, f"training_{args.qubits}q.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
