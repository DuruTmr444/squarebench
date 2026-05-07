"""
Two-figure visualisation of the pre-training benchmark runs.

Figure 1  — Purpose 1: Benchmark suite coverage and Qiskit transpilation cost
             (0001_qiskit.json)
    Panel A: log–log scatter of input qubits vs transpilation time,
             coloured by circuit family, to show the breadth of coverage.
    Panel B: number of benchmark records, median transpilation time, and
             median 2Q gate count per family as a summary bar chart.

Figure 2  — Purpose 2: Backend target comparison (0002–0010)
    Panel A: grouped bars — 133q HH, 156q HH, and 120q S /
             Nighthawk-style Qiskit preset transpilation results.
    Panel B: lollipop chart — percentage change vs the 133q HH run
             (negative = lower count/depth than 133q HH).

Usage:
    python plotting/plot_presummit_baseline.py
    python plotting/plot_presummit_baseline.py --fig 1
    python plotting/plot_presummit_baseline.py --fig 2
    python plotting/plot_presummit_baseline.py --fig 2 --metric depth
    python plotting/plot_presummit_baseline.py --fig 2 --metric both
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import next_plot_path

BENCHMARKS_DIR = ".benchmarks/Darwin-CPython-3.14-64bit"
RESULTS_PATH   = "./results"

# ── colour palette ──────────────────────────────────────────────────────────

FAMILY_META = {
    "hamiltonians":  {"label": "Hamiltonians (chemistry / opt.)", "color": "#5C6BC0"},
    "qasmbench":     {"label": "QASMBench circuits",              "color": "#26A69A"},
    "hamlib":        {"label": "HamLib Hamiltonians",             "color": "#FFA726"},
    "feynman":       {"label": "Feynman circuits",                "color": "#EF5350"},
    "device_100q":   {"label": "100Q device circuits",            "color": "#AB47BC"},
    "build_utils":   {"label": "Build / utility",                 "color": "#BDBDBD"},
}

TORINO_COLOR    = "#E53935"
MARRAKESH_COLOR = "#8E24AA"
MIAMI_COLOR     = "#1E88E5"

BACKEND_GROUPS = {
    "torino": {
        "label": "133q HH (run 02)",
        "short_label": "133q HH",
        "color": TORINO_COLOR,
        "files": ["0002_qiskit-test-summit.json"],
    },
    "marrakesh": {
        "label": "156q HH (runs 03-06)",
        "short_label": "156q HH",
        "color": MARRAKESH_COLOR,
        "files": [
            "0003_qiskit-test-summit.json",
            "0004_qiskit-test-summit.json",
            "0005_qiskit-test-summit.json",
            "0006_qiskit-test-summit.json",
        ],
    },
    "miami": {
        "label": "120q S (runs 09-10)",
        "short_label": "120q S",
        "color": MIAMI_COLOR,
        "files": [
            "0009_qiskit-test-summit-miami.json",
            "0010_qiskit-test-summit-miami.json",
        ],
    },
}

METRICS = {
    "count": {
        "key": "output_gate_count_2q",
        "ylabel": "Output 2Q gate count  [lower is better]",
        "title": "Qiskit Output 2Q Gate Count per Circuit",
        "lollipop_title": "2Q Gate Count Relative to 133q HH Run",
        "change_label": "% change vs 133q HH  [negative = fewer gates]",
        "stem": "2q_count",
    },
    "depth": {
        "key": "output_depth_2q",
        "ylabel": "Output 2Q gate depth  [lower is better]",
        "title": "Qiskit Output 2Q Gate Depth per Circuit",
        "lollipop_title": "2Q Gate Depth Relative to 133q HH Run",
        "change_label": "% change vs 133q HH  [negative = shallower]",
        "stem": "2q_depth",
    },
}

# ── circuit display order / labels ─────────────────────────────────────

CORE_ORDER = [
    "test_BV_100_transpile",
    "test_square_heisenberg_100_transpile",
    "test_circSU2_89_transpile",
    "test_circSU2_100_transpile",
    "test_QAOA_100_transpile",
    "test_QFT_100_transpile",
    "test_QV_100_transpile",
    "test_clifford_100_transpile",
]
CORE_SHORT = {
    "test_BV_100_transpile":                "BV",
    "test_square_heisenberg_100_transpile":  "Heisenberg",
    "test_circSU2_89_transpile":             "SU2-89",
    "test_circSU2_100_transpile":            "SU2-100",
    "test_QAOA_100_transpile":               "QAOA",
    "test_QFT_100_transpile":                "QFT",
    "test_QV_100_transpile":                 "QV",
    "test_clifford_100_transpile":           "Clifford",
}


# ── data loaders ────────────────────────────────────────────────────────────

def _load(filename):
    path = os.path.join(BENCHMARKS_DIR, filename)
    with open(path) as f:
        return json.load(f)["benchmarks"]


def _classify(name):
    """Map a benchmark name to a family key."""
    if name.startswith("test_hamiltonians"):
        return "hamiltonians"
    if name.startswith("test_QASMBench"):
        return "qasmbench"
    if name.startswith("test_hamlib"):
        return "hamlib"
    if name.startswith("test_feynman"):
        return "feynman"
    if name in CORE_SHORT:
        return "device_100q"
    return "build_utils"


def load_baseline():
    benchmarks = _load("0001_qiskit.json")
    records = []
    for b in benchmarks:
        ex  = b.get("extra_info", {})
        qubits = (
            ex.get("input_num_qubits")
            or ex.get("ham_qubits")
        )
        if b["name"] == "test_QV_100_transpile":
            qubits = 100
        records.append({
            "name":   b["name"],
            "family": _classify(b["name"]),
            "time":   b["stats"]["mean"],
            "qubits": qubits,
            "2q":     ex.get("output_gate_count_2q"),
        })
    return records


def load_backend_runs(metric):
    """Return backend_group -> test_name -> list of metric values across runs."""
    metric_key = METRICS[metric]["key"]

    def _collect(files):
        result = {}
        for fname in files:
            for b in _load(fname):
                name = b["name"]
                val = b.get("extra_info", {}).get(metric_key)
                if val is not None:
                    result.setdefault(name, []).append(float(val))
        return result

    return {
        group: _collect(meta["files"])
        for group, meta in BACKEND_GROUPS.items()
    }


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def plot_figure1(records, scatter_only=False):
    families = [k for k in FAMILY_META if k != "build_utils"]
    if scatter_only:
        fig, ax_scatter = plt.subplots(figsize=(10, 7))
    else:
        fig, (ax_scatter, ax_bars) = plt.subplots(
            1, 2, figsize=(16, 7),
            gridspec_kw={"width_ratios": [2, 1]},
        )
    fig.suptitle(
        "Benchpress Suite Run with Qiskit Transpilation Cost - 1,011 benchmarks",
        fontsize=13, fontweight="bold",
    )

    # scatter
    for family in FAMILY_META:
        if family == "build_utils":
            continue
        pts = [r for r in records if r["family"] == family and r["qubits"] and r["time"]]
        if not pts:
            continue
        xs = [r["qubits"] for r in pts]
        ys = [r["time"]   for r in pts]
        ax_scatter.scatter(
            xs, ys,
            color=FAMILY_META[family]["color"],
            label=f"{FAMILY_META[family]['label']} ({len(pts)} records)",
            alpha=0.55, s=18, linewidths=0,
        )

    # highlight the test-summit circuits selected for evaluation
    summit_pts = [r for r in records if r["name"] in CORE_SHORT and r["qubits"] and r["time"]]
    if summit_pts:
        ax_scatter.scatter(
            [r["qubits"] for r in summit_pts],
            [r["time"]   for r in summit_pts],
            color=FAMILY_META["device_100q"]["color"],
            s=160, marker="*", zorder=5,
            edgecolors="black", linewidths=0.5,
            label=f"Test-summit circuits ({len(summit_pts)} selected for evaluation)",
        )
        for r in summit_pts:
            ax_scatter.annotate(
                CORE_SHORT[r["name"]],
                (r["qubits"], r["time"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7.5, fontweight="bold",
                color=FAMILY_META["device_100q"]["color"],
            )

    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Input circuit qubits", fontsize=11)
    ax_scatter.set_ylabel("Transpilation time (s)", fontsize=11)
    ax_scatter.set_title("Coverage view: circuit size vs. transpilation time (log-log)", fontsize=11)
    ax_scatter.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_scatter.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(fontsize=8, loc="upper left", framealpha=0.9)

    if not scatter_only:
        counts = {}
        for r in records:
            counts[r["family"]] = counts.get(r["family"], 0) + 1

        fam_order  = [f for f in families if f in counts]
        fam_labels = [FAMILY_META[f]["label"].replace(" (", "\n(") for f in fam_order]
        fam_colors = [FAMILY_META[f]["color"] for f in fam_order]
        med_times  = []
        med_2q     = []
        for f in fam_order:
            pts = [r for r in records if r["family"] == f]
            med_times.append(np.median([r["time"] for r in pts]))
            valid_2q = [r["2q"] for r in pts if r["2q"] is not None]
            med_2q.append(np.median(valid_2q) if valid_2q else 0)

        x    = np.arange(len(fam_order))
        bars = ax_bars.bar(x, med_times, color=fam_colors, edgecolor="white",
                           linewidth=0.5, alpha=0.85)
        for bar, mt, m2q in zip(bars, med_times, med_2q):
            ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mt:.3f}s\n{int(m2q) if m2q else '—'} 2Q",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(fam_labels, fontsize=7.5, rotation=10, ha="right")
        ax_bars.set_ylabel("Median transpilation time (s)", fontsize=11)
        ax_bars.set_title(
            "Per-family cost summary\n(median time · median 2Q gate count)",
            fontsize=11,
        )
        ax_bars.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax_bars.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = next_plot_path(RESULTS_PATH, "presummit_baseline_fig1")
    plt.savefig(out, dpi=150)
    print(f"Figure 1 saved → {out}")
    plt.show()


def _lollipop_pct_marker(ax, x, ref_vals, cmp_vals, color, label, marker="o"):
    """Lollipop % difference with configurable marker shape."""
    ref_vals = np.asarray(ref_vals, dtype=float)
    cmp_vals = np.asarray(cmp_vals, dtype=float)
    pct = np.full_like(cmp_vals, np.nan, dtype=float)
    valid = (~np.isnan(ref_vals)) & (~np.isnan(cmp_vals)) & (ref_vals != 0)
    pct[valid] = (cmp_vals[valid] - ref_vals[valid]) / ref_vals[valid] * 100
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plotted_label = False
    for xi, v in zip(x, pct):
        if np.isnan(v):
            continue
        stem_color = "#4CAF50" if v < 0 else "#F44336"
        ax.vlines(xi, 0, v, colors=stem_color, linewidth=1.5)
        ax.plot(xi, v, marker, color=color, markersize=8, zorder=3,
                label=label if not plotted_label else "")
        plotted_label = True
        va = "top" if v < 0 else "bottom"
        ax.text(xi, v + (-1.5 if v < 0 else 1.5), f"{v:+.1f}%",
                ha="center", va=va, fontsize=7, fontweight="bold", color=stem_color)


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def plot_figure2(group_data, metric):
    metric_meta = METRICS[metric]
    group_names = list(BACKEND_GROUPS)
    circuits = [
        c for c in CORE_ORDER
        if all(c in group_data[group] for group in group_names)
    ]
    labels   = [CORE_SHORT[c] for c in circuits]
    x        = np.arange(len(circuits))

    means = {
        group: np.array([np.mean(group_data[group][c]) for c in circuits])
        for group in group_names
    }

    fig, (ax_bars, ax_lollipop) = plt.subplots(
        2, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle(
        "Qiskit Transpilation - Backend Comparison between Heron and Nighthawk\n"
        "133q HH vs 156q HH vs 120q S targets",
        fontsize=13, fontweight="bold",
    )

    # ── Panel A: grouped bars (log scale) ──────────────────────────────────
    w = 0.25
    offsets = {
        group: (idx - (len(group_names) - 1) / 2) * w
        for idx, group in enumerate(group_names)
    }
    for group in group_names:
        meta = BACKEND_GROUPS[group]
        bars = ax_bars.bar(
            x + offsets[group], means[group], width=w,
            color=meta["color"], label=meta["label"],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, means[group]):
            ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.08,
                f"{int(val):,}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=meta["color"], rotation=90,
            )

    ax_bars.set_yscale("log")
    ax_bars.set_ylabel(metric_meta["ylabel"] + "  [log scale]", fontsize=11)
    ax_bars.set_title(metric_meta["title"], fontsize=11)
    ax_bars.yaxis.grid(True, linestyle="--", alpha=0.5, which="both")
    ax_bars.set_axisbelow(True)
    ax_bars.legend(
        handles=[
            mpatches.Patch(color=BACKEND_GROUPS[group]["color"], label=BACKEND_GROUPS[group]["label"])
            for group in group_names
        ],
        loc="upper left", fontsize=9,
    )

    # ── Panel B: lollipop % change vs 133q HH run ──────────────────────
    torino_means = means["torino"]
    _lollipop_pct_marker(
        ax_lollipop, x - 0.15,
        torino_means, means["marrakesh"],
        MARRAKESH_COLOR, BACKEND_GROUPS["marrakesh"]["short_label"], "s",
    )
    _lollipop_pct_marker(
        ax_lollipop, x + 0.15,
        torino_means, means["miami"],
        MIAMI_COLOR, BACKEND_GROUPS["miami"]["short_label"], "o",
    )
    ax_lollipop.set_ylabel(metric_meta["change_label"], fontsize=11)
    ax_lollipop.set_title(metric_meta["lollipop_title"], fontsize=11)
    ax_lollipop.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_lollipop.set_axisbelow(True)
    ax_lollipop.fill_between(
        [-0.5, len(circuits) - 0.5], 0,
        min(ax_lollipop.get_ylim()[0], -5),
        alpha=0.07, color="#4CAF50", zorder=0,
    )
    ax_lollipop.legend(
        handles=[
            mpatches.Patch(color="#4CAF50", label="Lower than 133q HH"),
            mpatches.Patch(color="#F44336", label="Higher than 133q HH"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=MARRAKESH_COLOR,
                       markersize=8, label=BACKEND_GROUPS["marrakesh"]["short_label"]),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MIAMI_COLOR,
                       markersize=8, label=BACKEND_GROUPS["miami"]["short_label"]),
        ],
        loc="upper left", fontsize=8,
    )

    ax_lollipop.set_xticks(x)
    ax_lollipop.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax_lollipop.set_xlabel("Circuit", fontsize=11)

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out = next_plot_path(RESULTS_PATH, f"presummit_backend_comparison_{metric_meta['stem']}")
    plt.savefig(out, dpi=150)
    print(f"Figure 2 saved → {out}")
    plt.show()


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", choices=["1", "1a", "2", "both"], default="both",
                        help="Which figure to produce: 1=both panels, 1a=scatter only, 2, both (default: both)")
    parser.add_argument("--metric", choices=["count", "depth", "both"], default="count",
                        help="Metric for figure 2: count, depth, or both (default: count)")
    args = parser.parse_args()

    if args.fig in ("1", "1a", "both"):
        records = load_baseline()
        plot_figure1(records, scatter_only=(args.fig == "1a"))

    if args.fig in ("2", "both"):
        metrics = ["count", "depth"] if args.metric == "both" else [args.metric]
        for metric in metrics:
            group_data = load_backend_runs(metric)
            plot_figure2(group_data, metric)


if __name__ == "__main__":
    main()
