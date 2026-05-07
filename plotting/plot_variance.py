"""
Plot variance (std dev) of 2Q gate count and depth across qubit sizes and methods.

Reads from a benchmark_miami CSV that includes std_2q_gates and std_2q_depth columns
(produced by evaluation/benchmark_miami.py).

Usage:
    python plotting/plot_variance.py
    python plotting/plot_variance.py --csv results/benchmark_miami_20260419_123456.csv
"""

import argparse
import csv
import glob
import os
from plot_utils import next_plot_path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = "./results"

METHODS = ["qiskit", "qiskit_hf", "sqr"]
COLORS  = {"qiskit": "#9C27B0", "qiskit_hf": "#FF9800", "sqr": "#4CAF50"}
LABELS  = {"qiskit": "Qiskit", "qiskit_hf": "Qiskit HF AI", "sqr": "SQR HF AI"}


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v != "" else float("nan")) if k not in ("method",) else v
                         for k, v in row.items()})
    return rows


def get(rows, n, method, key):
    match = [r for r in rows if int(r["num_qubits"]) == n and r["method"] == method]
    return match[0].get(key, float("nan")) if match else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to benchmark CSV. Defaults to most recent benchmark_miami_*.csv in results/")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        candidates = sorted(glob.glob(os.path.join(RESULTS_PATH, "benchmark_miami_*.csv")))
        if not candidates:
            print("No benchmark_miami_*.csv found in results/. Run evaluation/benchmark_miami.py first.")
            return
        csv_path = candidates[-1]

    print(f"Loading: {csv_path}")
    rows = load_csv(csv_path)

    # Check std columns exist
    if "std_2q_gates" not in rows[0]:
        print("ERROR: CSV does not contain std_2q_gates/std_2q_depth columns.")
        print("Re-run evaluation/benchmark_miami.py to generate an updated CSV.")
        return

    qubit_counts = sorted(set(int(r["num_qubits"]) for r in rows))
    x = np.arange(len(qubit_counts))
    w = 0.25
    n_methods = len(METHODS)

    num_samples = int(rows[0].get("num_samples", 0))
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(
        f"Variance Analysis — FakeMiami Linear Function Benchmark\n"
        f"(n={num_samples} samples per qubit size)",
        fontsize=13,
    )

    # --- Panel 1: Std dev of 2Q gate count ---
    ax = axes[0]
    for i, method in enumerate(METHODS):
        vals = [get(rows, n, method, "std_2q_gates") for n in qubit_counts]
        offset = (i - (n_methods - 1) / 2) * w
        ax.bar(x + offset, vals, width=w, color=COLORS[method],
               edgecolor="white", linewidth=0.5, label=LABELS[method])
    ax.set_ylabel("Std Dev — 2Q Gate Count")
    ax.set_title("Variability in 2Q Gate Count  [lower = more consistent]")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    # --- Panel 2: Std dev of 2Q depth ---
    ax = axes[1]
    for i, method in enumerate(METHODS):
        vals = [get(rows, n, method, "std_2q_depth") for n in qubit_counts]
        offset = (i - (n_methods - 1) / 2) * w
        ax.bar(x + offset, vals, width=w, color=COLORS[method],
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Std Dev — 2Q Gate Depth")
    ax.set_title("Variability in 2Q Gate Depth  [lower = more consistent]")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # --- Panel 3: Coefficient of variation (std/mean) — normalised variability ---
    ax = axes[2]
    for i, method in enumerate(METHODS):
        cvs = []
        for n in qubit_counts:
            avg = get(rows, n, method, "avg_2q_gates")
            std = get(rows, n, method, "std_2q_gates")
            cvs.append((std / avg * 100) if avg and avg != 0 else float("nan"))
        offset = (i - (n_methods - 1) / 2) * w
        ax.bar(x + offset, cvs, width=w, color=COLORS[method],
               edgecolor="white", linewidth=0.5)
        for xi, cv in zip(x + offset, cvs):
            if not np.isnan(cv):
                ax.text(xi, cv + 0.2, f"{cv:.1f}%", ha="center", va="bottom",
                        fontsize=6, color="black")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Normalised Variability (std/mean)  [lower = more predictable]")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}q" for n in qubit_counts])
    ax.set_xlabel("Number of Qubits")

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = next_plot_path(RESULTS_PATH, "variance")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
