"""
Visualise linear-function synthesis benchmark results from a CSV file
produced by evaluation/benchmark_miami.py.

Usage:
    python plotting/plot_random_lf_circuits.py
    python plotting/plot_random_lf_circuits.py --csv results/benchmark_miami_20260419_235800.csv
    python plotting/plot_random_lf_circuits.py --csv results/benchmark_marrakesh_20260419_235800.csv
    python plotting/plot_random_lf_circuits.py --csv results/random_linear_functions/benchmark_miami_5.csv --panels bars
    python plotting/plot_random_lf_circuits.py --csv results/random_linear_functions/benchmark_miami_5.csv --panels lollipop
    python plotting/plot_random_lf_circuits.py --csv results/random_linear_functions/benchmark_miami_5.csv --panels depth
    python plotting/plot_random_lf_circuits.py --csv results/random_linear_functions/benchmark_miami_5.csv --panels scatter
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import METHODS, COLORS, LABELS, bar_group, lollipop_pct, next_plot_path

RESULTS_PATH = "./results"
RANDOM_LF_RESULTS_PATH = "./results/random_linear_functions"


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method == "duru":
                method = "sqr"
            rows.append({
                "num_qubits":   int(row["num_qubits"]),
                "method":       method,
                "avg_2q_gates": float(row["avg_2q_gates"]),
                "std_2q_gates": float(row["std_2q_gates"]) if "std_2q_gates" in row else float("nan"),
                "avg_2q_depth": float(row["avg_2q_depth"]),
                "std_2q_depth": float(row["std_2q_depth"]) if "std_2q_depth" in row else float("nan"),
            })
    return rows


def _backend_label(backend_name):
    return {
        "miami":     "FakeMiami (Square Lattice)",
        "marrakesh": "FakeMarrakesh (Heavy-Hex)",
    }.get(backend_name, backend_name)


def _plot_bar_panels(axes, rows, qubit_counts, x):
    for ax, avg_metric, std_metric, ylabel in [
        (axes[0], "avg_2q_gates", "std_2q_gates", "Avg 2Q Gate Count"),
        (axes[1], "avg_2q_depth", "std_2q_depth", "Avg 2Q Gate Depth"),
    ]:
        vals_by_method, errs_by_method = {}, {}
        for method in METHODS:
            vals, errs = [], []
            for n in qubit_counts:
                match = next((r for r in rows if r["num_qubits"] == n and r["method"] == method), None)
                vals.append(match[avg_metric] if match else float("nan"))
                errs.append(match[std_metric] if match else float("nan"))
            vals_by_method[method] = vals
            errs_by_method[method] = errs
        bar_group(ax, x, vals_by_method, errs_by_method=errs_by_method)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left")


def _plot_lollipop_panels(axes, rows, qubit_counts, x):
    for ax, metric, ylabel in [
        (axes[0], "avg_2q_gates", "% Diff vs Qiskit HF AI\n(2Q Gate Count)"),
        (axes[1], "avg_2q_depth", "% Diff vs Qiskit HF AI\n(2Q Gate Depth)"),
    ]:
        sqr_vals = np.array([
            next((r[metric] for r in rows if r["num_qubits"] == n and r["method"] == "sqr"), float("nan"))
            for n in qubit_counts
        ])
        hf_vals = np.array([
            next((r[metric] for r in rows if r["num_qubits"] == n and r["method"] == "qiskit_hf"), float("nan"))
            for n in qubit_counts
        ])
        lollipop_pct(ax, x, hf_vals, sqr_vals, COLORS["sqr"], LABELS["sqr"])
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.fill_between([-0.5, len(qubit_counts) - 0.5], 0,
                        min(ax.get_ylim()[0], -1), alpha=0.06, color="#4CAF50", zorder=0)
        ax.legend(
            handles=[
                Patch(color="#4CAF50", label="SQR better"),
                Patch(color="#F44336", label="SQR worse"),
            ],
            loc="upper left", fontsize=8,
        )


def _plot_depth_panels(axes, rows, qubit_counts, x):
    vals_by_method, errs_by_method = {}, {}
    for method in METHODS:
        vals, errs = [], []
        for n in qubit_counts:
            match = next((r for r in rows if r["num_qubits"] == n and r["method"] == method), None)
            vals.append(match["avg_2q_depth"] if match else float("nan"))
            errs.append(match["std_2q_depth"] if match else float("nan"))
        vals_by_method[method] = vals
        errs_by_method[method] = errs

    bar_group(axes[0], x, vals_by_method, errs_by_method=errs_by_method)
    axes[0].set_ylabel("Avg 2Q Gate Depth")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper left")

    sqr_vals = np.array([
        next((r["avg_2q_depth"] for r in rows if r["num_qubits"] == n and r["method"] == "sqr"), float("nan"))
        for n in qubit_counts
    ])
    hf_vals = np.array([
        next((r["avg_2q_depth"] for r in rows if r["num_qubits"] == n and r["method"] == "qiskit_hf"), float("nan"))
        for n in qubit_counts
    ])
    lollipop_pct(axes[1], x, hf_vals, sqr_vals, COLORS["sqr"], LABELS["sqr"])
    axes[1].set_ylabel("% Diff vs Qiskit HF AI\n(2Q Gate Depth)")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)
    axes[1].set_axisbelow(True)
    axes[1].fill_between([-0.5, len(qubit_counts) - 0.5], 0,
                         min(axes[1].get_ylim()[0], -1), alpha=0.06, color="#4CAF50", zorder=0)
    axes[1].legend(
        handles=[
            Patch(color="#4CAF50", label="SQR better"),
            Patch(color="#F44336", label="SQR worse"),
        ],
        loc="upper left", fontsize=8,
    )


def _scatter_panel(ax, rows, qubit_counts, avg_metric, std_metric, xlabel, ylabel, title):
    markers = {"qiskit_hf": "s", "sqr": "o"}
    label_offset = {"qiskit_hf": (5, 4), "sqr": (5, -10)}

    vals, errs = {}, {}
    for method in METHODS:
        v, e = [], []
        for n in qubit_counts:
            r = next((row for row in rows if row["num_qubits"] == n and row["method"] == method), None)
            v.append(r[avg_metric] if r else float("nan"))
            e.append(r[std_metric] if r else float("nan"))
        vals[method] = np.array(v)
        errs[method] = np.array(e)

    ref = vals["qiskit"]
    all_vals = np.concatenate([vals[m][~np.isnan(vals[m])] for m in METHODS])
    lo = all_vals.min() * 0.88
    hi = all_vals.max() * 1.08

    diag = np.array([lo, hi])
    ax.fill_between(diag, lo, diag, alpha=0.07, color="#4CAF50", zorder=0)
    ax.fill_between(diag, diag, hi, alpha=0.07, color="#F44336", zorder=0)
    ax.plot(diag, diag, "--", color=COLORS["qiskit"], linewidth=1.5, zorder=1)

    has_errs = not all(np.all(np.isnan(errs[m])) for m in ["qiskit_hf", "sqr"])

    for cmp_method in ["qiskit_hf", "sqr"]:
        for i, n in enumerate(qubit_counts):
            xv, yv = ref[i], vals[cmp_method][i]
            if np.isnan(xv) or np.isnan(yv):
                continue
            if has_errs:
                ax.errorbar(xv, yv, xerr=errs["qiskit"][i], yerr=errs[cmp_method][i],
                            fmt="none", ecolor="grey", elinewidth=1, capsize=3, zorder=2)
            ax.scatter(xv, yv, color=COLORS[cmp_method], s=90, zorder=4,
                       marker=markers[cmp_method], edgecolors="white", linewidths=0.6)
            ox, oy = label_offset[cmp_method]
            ax.annotate(f"{n}q", (xv, yv), textcoords="offset points",
                        xytext=(ox, oy), fontsize=7, fontweight="bold",
                        color=COLORS[cmp_method])

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], linestyle="--", color=COLORS["qiskit"], linewidth=1.5, label="Qiskit"),
            plt.Line2D([0], [0], linestyle="none", marker="s", color=COLORS["qiskit_hf"],
                       markersize=8, label=LABELS["qiskit_hf"]),
            plt.Line2D([0], [0], linestyle="none", marker="o", color=COLORS["sqr"],
                       markersize=8, label=LABELS["sqr"]),
            Patch(color="#4CAF50", label="Better"),
            Patch(color="#F44336", label="Worse"),
        ],
        fontsize=8, loc="upper left",
    )


def _plot_scatter_panels(axes, rows, qubit_counts):
    _scatter_panel(axes[0], rows, qubit_counts,
                   "avg_2q_gates", "std_2q_gates",
                   "Qiskit  avg 2Q gates", "avg 2Q gates", "2Q Gate Count")
    _scatter_panel(axes[1], rows, qubit_counts,
                   "avg_2q_depth", "std_2q_depth",
                   "Qiskit  avg 2Q depth", "avg 2Q depth", "2Q Gate Depth")


def plot(rows, backend_name="miami", panels="all"):
    qubit_counts = sorted(set(r["num_qubits"] for r in rows))
    x = np.arange(len(qubit_counts))

    if panels == "scatter":
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        _plot_scatter_panels(axes, rows, qubit_counts)
        stem = f"benchmark_{backend_name}_scatter"
        fig.suptitle(
            f"Gate Quality vs Qiskit Baseline  |  {_backend_label(backend_name)}\n"
            "Linear Function Synthesis  [each point = one qubit count, diagonal = Qiskit]",
            fontsize=13,
        )
        plt.tight_layout()
    elif panels == "scatter-depth":
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _scatter_panel(ax, rows, qubit_counts,
                       "avg_2q_depth", "std_2q_depth",
                       "Qiskit  avg 2Q depth", "avg 2Q depth", "2Q Gate Depth")
        stem = f"benchmark_{backend_name}_scatter_depth"
        fig.suptitle(
            f"2Q Gate Depth vs Qiskit  |  {_backend_label(backend_name)}\n"
            "Linear Function Synthesis  [diagonal = Qiskit]",
            fontsize=12,
        )
        plt.tight_layout()
    else:
        n_panels = 4 if panels == "all" else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 14 if panels == "all" else 8), sharex=True)

        if panels == "bars":
            _plot_bar_panels(axes, rows, qubit_counts, x)
            stem = f"benchmark_{backend_name}_bars"
        elif panels == "lollipop":
            _plot_lollipop_panels(axes, rows, qubit_counts, x)
            stem = f"benchmark_{backend_name}_lollipop"
        elif panels == "depth":
            _plot_depth_panels(axes, rows, qubit_counts, x)
            stem = f"benchmark_{backend_name}_depth"
        else:
            _plot_bar_panels(axes[:2], rows, qubit_counts, x)
            _plot_lollipop_panels(axes[2:], rows, qubit_counts, x)
            stem = f"benchmark_{backend_name}"

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"{n}q" for n in qubit_counts])
        axes[-1].set_xlabel("Number of Qubits")
        fig.suptitle(
            f"Linear Function Synthesis on {_backend_label(backend_name)}\n"
            "Qiskit vs Qiskit HF AI vs SQR HF AI"
            + (" — 2Q Depth" if panels == "depth" else ""),
            fontsize=13,
        )
        plt.tight_layout()

    os.makedirs(RESULTS_PATH, exist_ok=True)
    out_dir = RANDOM_LF_RESULTS_PATH if panels == "scatter" else RESULTS_PATH
    os.makedirs(out_dir, exist_ok=True)
    fig_path = next_plot_path(out_dir, stem)
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to benchmark CSV (default: most recent benchmark_*.csv in results/)")
    parser.add_argument("--backend", default=None, choices=["miami", "marrakesh"],
                        help="Backend name for plot title (inferred from filename if omitted)")
    parser.add_argument("--panels", default="all",
                        choices=["all", "bars", "lollipop", "depth", "scatter", "scatter-depth"],
                        help="Which panels to plot")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        files = sorted(
            glob.glob(os.path.join(RESULTS_PATH, "benchmark_*.csv"))
            + glob.glob(os.path.join(RANDOM_LF_RESULTS_PATH, "benchmark_*.csv"))
        )
        if not files:
            print("No benchmark CSV found in results/. Run evaluation/benchmark_miami.py first.")
            return
        csv_path = files[-1]
    print(f"Loading: {csv_path}")

    backend = args.backend
    if backend is None:
        backend = "marrakesh" if "marrakesh" in os.path.basename(csv_path) else "miami"

    rows = load_csv(csv_path)
    plot(rows, backend_name=backend, panels=args.panels)


if __name__ == "__main__":
    main()
