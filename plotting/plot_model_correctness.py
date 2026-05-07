"""
Plot post-training correctness evaluation results from evaluate_square_lattice.py.

Reads the most recent results/eval_*.csv and produces a three-panel figure:
  - Panel 1: success rate per model (output == input up to linear equivalence)
  - Panel 2: average CNOT count (successful syntheses only)
  - Panel 3: average CNOT depth (successful syntheses only)

Bars are colour-coded by qubit count (4–6: teal, 7: orange, 8: blue).

Usage:
    python plotting/plot_model_correctness.py
    python plotting/plot_model_correctness.py --csv results/eval_20260419_123456.csv
"""

import argparse
import glob
import json
import os
import sys
from plot_utils import next_plot_path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

RESULTS_PATH = "./results"

QUBIT_COLORS = {
    4: "#2A9D8F", 5: "#2A9D8F", 6: "#2A9D8F",
    7: "#F4A261",
    8: "#4C9BE8",
    9: "#E76F51",
    10: "#9B5DE5",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to evaluation CSV (default: most recent results/eval_*.csv)")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        files = sorted(glob.glob(os.path.join(RESULTS_PATH, "eval_*.csv")))
        if not files:
            print("No evaluation CSV found in results/. Run evaluation/evaluate_square_lattice.py first.")
            return
        csv_path = files[-1]
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values(["num_qubits", "name"]).reset_index(drop=True)
    short_names = df["name"].str.replace("linear_function_", "", regex=False)

    meta_path = csv_path.replace(".csv", "_meta.json")
    duration_label = ""
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        mins = meta["duration_seconds"] / 60
        duration_label = f"  |  Total evaluation time: {mins:.1f} min"

    colors = df["num_qubits"].map(lambda n: QUBIT_COLORS.get(n, "#999999"))
    unique_qs = sorted(df["num_qubits"].unique())
    legend_elements = [Patch(facecolor=QUBIT_COLORS.get(q, "#999"), label=f"{q} qubits") for q in unique_qs]

    x = range(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(f"Post-Training Correctness Evaluation{duration_label}", fontsize=13)

    axes[0].bar(x, df["success_rate"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title(os.path.basename(csv_path))
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0, 105)
    axes[0].legend(handles=legend_elements, loc="lower right")

    axes[1].bar(x, df["avg_cnots"], color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Avg CNOT Count")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[1].set_axisbelow(True)

    axes[2].bar(x, df["avg_cnot_depth"], color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_ylabel("Avg CNOT Depth")
    axes[2].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[2].set_axisbelow(True)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(short_names, rotation=90, fontsize=8)

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = next_plot_path(RESULTS_PATH, "model_correctness")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
