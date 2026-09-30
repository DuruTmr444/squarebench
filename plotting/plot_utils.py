"""
Shared plotting primitives for synthesis benchmark visualisations.
"""

import os
import numpy as np


def next_plot_path(directory, stem, ext="png"):
    """Return the next available path for an output file.

    The first save uses ``<stem>.<ext>``, subsequent saves use
    ``<stem>_2.<ext>``, ``<stem>_3.<ext>``, etc.
    """
    candidate = os.path.join(directory, f"{stem}.{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(directory, f"{stem}_{n}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1

METHODS = ["qiskit", "qiskit_hf", "sqr"]
COLORS  = {"qiskit": "#9C27B0", "qiskit_hf": "#FF9800", "sqr": "#4CAF50"}
LABELS  = {"qiskit": "Qiskit",  "qiskit_hf": "QTS", "sqr": "SQR"}


def bar_group(ax, x, vals_by_method, w=0.25, errs_by_method=None, log=False):
    """Grouped bar chart for each method. Optionally adds error bars and log scale."""
    n = len(METHODS)
    for i, method in enumerate(METHODS):
        vals = vals_by_method[method]
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, vals, width=w, color=COLORS[method],
               edgecolor="white", linewidth=0.5, label=LABELS[method])
        if errs_by_method and method in errs_by_method:
            ax.errorbar(x + offset, vals, yerr=errs_by_method[method],
                        fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=5)
        for xi, v in zip(x + offset, vals):
            if np.isnan(float(v)):
                ax.text(xi, 0.5 if not log else 1, "N/A",
                        ha="center", va="bottom", fontsize=9, color="grey", rotation=90)
            else:
                ax.text(xi, float(v), f"{int(v)}" if v == int(v) else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold", color="black", rotation=90)
    if log:
        ax.set_yscale("log")


def lollipop_pct(ax, x, ref_vals, cmp_vals, color, label):
    """Lollipop chart of percentage difference of cmp_vals relative to ref_vals.
    Negative values (cmp < ref) are coloured green; positive values red."""
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
        ax.vlines(xi, 0, v, colors=stem_color, linewidth=2)
        ax.plot(xi, v, "o", color=color, markersize=8, zorder=3,
                label=label if not plotted_label else "")
        plotted_label = True
        va = "top" if v < 0 else "bottom"
        offset_y = -1.5 if v < 0 else 1.5
        ax.text(xi, v + offset_y, f"{v:+.1f}%", ha="center", va=va,
                fontsize=9, fontweight="bold", color=stem_color)
    ymin, ymax = ax.get_ylim()
    y_range = max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - 0.05 * y_range, ymax + 0.15 * y_range)
