"""
Plot SQR summit benchmark time vs Qiskit preset PM.

Usage:
    python plotting/plot_sqr_times.py
    python plotting/plot_sqr_time.py --sqr results/sqr_summit.json --qiskit results/qiskit.json
"""

import argparse
import json
import os
from plot_utils import METHODS, COLORS, LABELS, bar_group, lollipop_pct, next_plot_path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = "./results"

TEST_ORDER = [
    "test_BVlike_simplification_transpile",
    "test_square_heisenberg_100_transpile",
    "test_circSU2_100_transpile",
    "test_BV_100_transpile",
    "test_circSU2_89_transpile",
    "test_QAOA_100_transpile",
    "test_QFT_100_transpile",
    "test_QV_100_transpile",
]
TEST_SHORT = {
    "test_BVlike_simplification_transpile":  "BVlike",
    "test_square_heisenberg_100_transpile":  "Heisenberg",
    "test_circSU2_100_transpile":            "SU2-100",
    "test_BV_100_transpile":                 "BV",
    "test_circSU2_89_transpile":             "SU2-89",
    "test_QAOA_100_transpile":               "QAOA",
    "test_QFT_100_transpile":                "QFT",
    "test_QV_100_transpile":                 "QV",
}

COLORS = {"qiskit": "#9C27B0", "qiskit_hf": "#FF9800", "sqr": "#4CAF50"}
LABELS = {"qiskit": "Qiskit (preset PM)", "qiskit_hf": "Qiskit HF AI", "sqr": "SQR HF AI"}


def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for b in data.get("benchmarks", []):
        result[b["name"]] = {
            "time":  b["stats"]["mean"],
            "2q":    b["extra_info"].get("output_gate_count_2q"),
            "depth": b["extra_info"].get("output_depth_2q"),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqr",       default="results/sqr_summit.json")
    parser.add_argument("--qiskit",    default="results/qiskit.json")
    parser.add_argument("--qiskit-hf", default="results/qiskit_hf_summit.json")
    args = parser.parse_args()

    sqr       = load_json(args.sqr)
    qiskit    = load_json(args.qiskit)
    qiskit_hf = load_json(args.qiskit_hf)

    tests  = [t for t in TEST_ORDER if t in sqr]
    labels = [TEST_SHORT[t] for t in tests]
    x      = np.arange(len(tests))
    w      = 0.25

    sqr_times       = np.array([sqr[t]["time"]                         for t in tests])
    qiskit_times    = np.array([qiskit.get(t, {}).get("time", np.nan)  for t in tests])
    qiskit_hf_times = np.array([qiskit_hf.get(t, {}).get("time", np.nan) for t in tests])

    has_qiskit_hf = not all(np.isnan(qiskit_hf_times))

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    fig.suptitle(
        "SQR HF AI vs Qiskit vs Qiskit HF AI — FakeMiami Summit Benchmark",
        fontsize=14,
    )

    # --- Panel 1: Transpile time (log scale) ---
    ax = axes[0]
    offsets = [-w, 0, w] if has_qiskit_hf else [-w / 2, w / 2]
    bars = [
        (qiskit_times,    COLORS["qiskit"],    LABELS["qiskit"]),
        (sqr_times,       COLORS["sqr"],       LABELS["sqr"]),
    ]
    if has_qiskit_hf:
        bars.insert(1, (qiskit_hf_times, COLORS["qiskit_hf"], LABELS["qiskit_hf"]))

    for (vals, color, label), offset in zip(bars, offsets):
        ax.bar(x + offset, vals, width=w, color=color, edgecolor="white", label=label)
        for xi, v in zip(x + offset, vals):
            if np.isnan(v):
                ax.text(xi, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-3,
                        "N/A", ha="center", va="bottom", fontsize=6, color="grey", rotation=90)

    ax.set_yscale("log")
    ax.set_ylabel("Transpile time (s) [log scale]")
    ax.set_title("Transpile Time")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # --- Panel 2: Speedup lollipop (SQR vs Qiskit preset) ---
    ax = axes[1]
    speedup = qiskit_times / sqr_times  # >1 = SQR faster
    valid   = speedup[~np.isnan(speedup)]
    ymax    = max(valid) * 1.15 if len(valid) else 2

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between([-0.5, len(tests) - 0.5], 1, ymax, alpha=0.05, color="#4CAF50", zorder=0)

    for xi, s in zip(x, speedup):
        if np.isnan(s):
            ax.text(xi, 1.05, "N/A", ha="center", va="bottom", fontsize=7, color="grey")
            continue
        color = "#4CAF50" if s >= 1 else "#F44336"
        ax.vlines(xi, 1, s, colors=color, linewidth=2.5)
        ax.plot(xi, s, "o", color=color, markersize=10, zorder=3)
        label_txt = f"{s:.1f}x faster" if s >= 1 else f"{1/s:.1f}x slower"
        va = "bottom" if s >= 1 else "top"
        offset = 0.04 * ymax if s >= 1 else -0.04 * ymax
        ax.text(xi, s + offset, label_txt, ha="center", va=va,
                fontsize=8, fontweight="bold", color=color)

    # Also overlay Qiskit HF speedup as diamonds if available
    if has_qiskit_hf:
        speedup_hf = qiskit_times / qiskit_hf_times
        for xi, s in zip(x, speedup_hf):
            if np.isnan(s):
                continue
            color = "#FF9800"
            ax.plot(xi, s, "D", color=color, markersize=7, zorder=4,
                    label=LABELS["qiskit_hf"] if xi == x[0] else "")
        ax.legend(handles=[
            mpatches.Patch(color="#4CAF50", label="SQR faster than Qiskit"),
            mpatches.Patch(color="#F44336", label="SQR slower than Qiskit"),
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#FF9800",
                       markersize=8, label="Qiskit HF AI speedup over Qiskit"),
        ], loc="upper right", fontsize=8)
    else:
        ax.legend(handles=[
            mpatches.Patch(color="#4CAF50", label="SQR faster"),
            mpatches.Patch(color="#F44336", label="SQR slower"),
        ], loc="upper right")

    ax.set_ylabel("Speedup vs Qiskit preset PM")
    ax.set_title("Speedup over Qiskit  [>1 = faster, circles=SQR, diamonds=Qiskit HF AI]")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Circuit")

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = next_plot_path(RESULTS_PATH, "sqr_summit")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
