"""
Plot 2Q gate count, 2Q depth, and outperformance across Qiskit, Qiskit HF AI, DuruTo HF AI
on the Benchpress circuit suite (Benchpress JSON output files).

Usage:
    python plotting/plot_benchpress_summit.py
    python plotting/plot_benchpress_summit.py --panels bars
    python plotting/plot_benchpress_summit.py --panels total
    python plotting/plot_benchpress_summit.py --panels quality
    python plotting/plot_benchpress_summit.py --duru benchpress/results/qasm_sqr2.json \
                                               --qiskit benchpress/results/qasm_qiskit2.json \
                                               --qiskit-hf benchpress/results/qasm_hf_ai2.json
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import METHODS, COLORS, LABELS, bar_group, lollipop_pct, next_plot_path

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
    "test_clifford_100_transpile",
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
    "test_clifford_100_transpile":           "Clifford",
}


def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for b in data.get("benchmarks", []):
        ops = b["extra_info"].get("output_circuit_operations") or {}
        total_gates = sum(ops.values()) if ops else None
        result[b["name"]] = {
            "time":        b["stats"]["mean"],
            "2q":          b["extra_info"].get("output_gate_count_2q"),
            "depth":       b["extra_info"].get("output_depth_2q"),
            "total_gates": total_gates,
        }
    return result


def get_vals(data, tests, key):
    vals = []
    for test in tests:
        value = data.get(test, {}).get(key)
        vals.append(np.nan if value is None else float(value))
    return np.array(vals)


def style_pct_axis(ax, tests):
    ax.fill_between([-0.5, len(tests) - 0.5], 0, min(ax.get_ylim()[0], -1),
                    alpha=0.05, color="#4CAF50", zorder=0)
    ax.set_ylabel("% diff vs Qiskit")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def set_x_axis(ax, x, labels):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("Circuit")


def plot_bar_panels(axes, data, tests, x, legend_patches):
    bar_group(axes[0], x, {m: get_vals(data[m], tests, "2q") for m in METHODS})
    axes[0].set_ylabel("2Q gate count")
    axes[0].set_title("Output 2Q Gate Count  [lower is better]")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_axisbelow(True)
    axes[0].legend(handles=legend_patches, loc="upper left")

    bar_group(axes[1], x, {m: get_vals(data[m], tests, "depth") for m in METHODS})
    axes[1].set_ylabel("2Q gate depth")
    axes[1].set_title("Output 2Q Gate Depth  [lower is better]")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.6)
    axes[1].set_axisbelow(True)


def plot_total_panel(ax, data, tests, x, pct_legend):
    ref_total = get_vals(data["qiskit"], tests, "total_gates")
    lollipop_pct(ax, x,        ref_total, get_vals(data["qiskit_hf"], tests, "total_gates"), COLORS["qiskit_hf"], LABELS["qiskit_hf"])
    lollipop_pct(ax, x + 0.15, ref_total, get_vals(data["sqr"],       tests, "total_gates"), COLORS["sqr"],       LABELS["sqr"])
    style_pct_axis(ax, tests)
    ax.set_title("Total Gate Count % Difference vs Qiskit  [negative = fewer gates]")
    ax.legend(handles=pct_legend, loc="upper left", fontsize=8)


def plot_quality_panels(axes, data, tests, x):
    ref_2q = get_vals(data["qiskit"], tests, "2q")
    lollipop_pct(axes[0], x,        ref_2q, get_vals(data["qiskit_hf"], tests, "2q"), COLORS["qiskit_hf"], LABELS["qiskit_hf"])
    lollipop_pct(axes[0], x + 0.15, ref_2q, get_vals(data["sqr"],       tests, "2q"), COLORS["sqr"],       LABELS["sqr"])
    style_pct_axis(axes[0], tests)
    axes[0].set_title("2Q Gate Count % Difference vs Qiskit  [negative = fewer gates]")

    ref_depth = get_vals(data["qiskit"], tests, "depth")
    lollipop_pct(axes[1], x,        ref_depth, get_vals(data["qiskit_hf"], tests, "depth"), COLORS["qiskit_hf"], LABELS["qiskit_hf"])
    lollipop_pct(axes[1], x + 0.15, ref_depth, get_vals(data["sqr"],       tests, "depth"), COLORS["sqr"],       LABELS["sqr"])
    style_pct_axis(axes[1], tests)
    axes[1].set_title("2Q Gate Depth % Difference vs Qiskit  [negative = shallower]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duru",      default="results/jsons/qasm_sqr_3.json")
    parser.add_argument("--qiskit",    default="results/jsons/qasm_qiskit_3.json")
    parser.add_argument("--qiskit-hf", default="results/jsons/qasm_qiskit_hf_3.json")
    parser.add_argument("--panels", choices=["all", "bars", "total", "quality", "depth"], default="all",
                        help="Which panels to plot: all, bars, total, quality, or depth (depth bar + depth lollipop)")
    args = parser.parse_args()

    data = {
        "qiskit":    load_json(args.qiskit),
        "qiskit_hf": load_json(args.qiskit_hf),
        "sqr":       load_json(args.duru),
    }

    tests = [t for t in TEST_ORDER
             if any(data[m].get(t, {}).get("total_gates") is not None for m in METHODS)]
    if not tests:
        print("No gate count data found in any result file.")
        return

    labels = [TEST_SHORT[t] for t in tests]
    x = np.arange(len(tests))

    panel_count = {"all": 5, "bars": 2, "total": 1, "quality": 2, "depth": 2}[args.panels]
    fig_height = {"all": 19, "bars": 8, "total": 6, "quality": 8, "depth": 8}[args.panels]
    fig, axes = plt.subplots(panel_count, 1, figsize=(13, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        "Circuit Quality Comparison — FakeMiami QASM Benchmark\n"
        "Qiskit vs Qiskit HF AI vs SQR HF AI",
        fontsize=14,
    )

    legend_patches = [mpatches.Patch(color=COLORS[m], label=LABELS[m]) for m in METHODS]
    pct_legend = [
        mpatches.Patch(color="#4CAF50", label="Better than Qiskit"),
        mpatches.Patch(color="#F44336", label="Worse than Qiskit"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["sqr"],
                   markersize=8, label=LABELS["sqr"]),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["qiskit_hf"],
                   markersize=8, label=LABELS["qiskit_hf"]),
    ]

    if args.panels == "bars":
        plot_bar_panels(axes, data, tests, x, legend_patches)
        stem = "benchpress_qasm_bars"
    elif args.panels == "total":
        plot_total_panel(axes[0], data, tests, x, pct_legend)
        stem = "benchpress_qasm_total"
    elif args.panels == "quality":
        plot_quality_panels(axes, data, tests, x)
        axes[0].legend(handles=pct_legend, loc="upper left", fontsize=8)
        stem = "benchpress_qasm_quality"
    elif args.panels == "depth":
        bar_group(axes[0], x, {m: get_vals(data[m], tests, "depth") for m in METHODS})
        axes[0].set_ylabel("2Q gate depth")
        axes[0].set_title("Output 2Q Gate Depth  [lower is better]")
        axes[0].yaxis.grid(True, linestyle="--", alpha=0.6)
        axes[0].set_axisbelow(True)
        axes[0].legend(handles=legend_patches, loc="upper left")

        ref_depth = get_vals(data["qiskit"], tests, "depth")
        lollipop_pct(axes[1], x,        ref_depth, get_vals(data["qiskit_hf"], tests, "depth"), COLORS["qiskit_hf"], LABELS["qiskit_hf"])
        lollipop_pct(axes[1], x + 0.15, ref_depth, get_vals(data["sqr"],       tests, "depth"), COLORS["sqr"],       LABELS["sqr"])
        style_pct_axis(axes[1], tests)
        axes[1].set_title("2Q Gate Depth % Difference vs Qiskit  [negative = shallower]")
        axes[1].legend(handles=pct_legend, loc="upper left", fontsize=8)
        stem = "benchpress_qasm_depth"
    else:
        plot_bar_panels(axes[:2], data, tests, x, legend_patches)
        plot_total_panel(axes[2], data, tests, x, pct_legend)
        plot_quality_panels(axes[3:], data, tests, x)
        stem = "benchpress_qasm"

    set_x_axis(axes[-1], x, labels)

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = next_plot_path(RESULTS_PATH, stem)
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
