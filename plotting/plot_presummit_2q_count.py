"""
Plot 2-qubit gate counts and depth of the INPUT circuits in test_summit.py (before transpilation).

Usage:
    python plotting/plot_presummit_2q_count.py
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit.circuit.library import EfficientSU2, QuantumVolume
from qiskit import qasm2
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS, LEGACY_INCLUDE_PATH

from benchpress.qiskit_gym.circuits import bv_all_ones

QASM_BASE   = os.path.join(os.path.dirname(__file__), "..", "benchpress", "qasm")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")

COLOR_COUNT = "#5B9BD5"
COLOR_DEPTH = "#ED7D31"


def load_qasm(path):
    return qasm2.load(
        path,
        include_path=LEGACY_INCLUDE_PATH,
        custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
    )


def count_2q(circuit):
    return circuit.decompose().num_nonlocal_gates()


def depth_2q(circuit):
    return circuit.decompose().depth(lambda gate: gate.operation.num_qubits == 2)


circuits = {
    "QFT":        lambda: load_qasm(os.path.join(QASM_BASE, "qft", "qft_N100.qasm")),
    "QV":         lambda: QuantumVolume(100, 100, seed=12345),
    "SU2-89":     lambda: EfficientSU2(89, reps=3, entanglement="circular"),
    "SU2-100":    lambda: EfficientSU2(100, reps=3, entanglement="circular"),
    "BV":         lambda: bv_all_ones(100),
    "Heisenberg": lambda: load_qasm(os.path.join(QASM_BASE, "square-heisenberg", "square_heisenberg_N100.qasm")),
    "QAOA":       lambda: load_qasm(os.path.join(QASM_BASE, "qaoa", "qaoa_barabasi_albert_N100_3reps.qasm")),
    "Clifford":   lambda: load_qasm(os.path.join(QASM_BASE, "clifford", "clifford_100_12345.qasm")),
}


def main():
    names, counts, depths = [], [], []
    for name, builder in circuits.items():
        print(f"Loading {name}...")
        qc = builder()
        n2q = count_2q(qc)
        d2q = depth_2q(qc)
        print(f"  {name}: {n2q} 2Q gates, depth {d2q}")
        names.append(name)
        counts.append(n2q)
        depths.append(d2q)

    # sort by 2Q gate count ascending
    order  = np.argsort(counts)
    names  = [names[i]  for i in order]
    counts = [counts[i] for i in order]
    depths = [depths[i] for i in order]

    x = np.arange(len(names))
    w = 0.38

    fig, ax1 = plt.subplots(figsize=(12, 5))

    bars1 = ax1.bar(x - w / 2, counts, width=w, color=COLOR_COUNT,
                    edgecolor="white", linewidth=0.5, label="2Q gate count")
    bars2 = ax1.bar(x + w / 2, depths, width=w, color=COLOR_DEPTH,
                    edgecolor="white", linewidth=0.5, label="2Q depth")

    top = max(max(counts), max(depths))
    for bar, v in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + top * 0.01,
                 str(v), ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=COLOR_COUNT)
    for bar, v in zip(bars2, depths):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + top * 0.01,
                 str(v), ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=COLOR_DEPTH)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("Count / depth")
    ax1.set_title(
        "Input Circuit 2Q Gate Count & Depth — test_summit.py  (before transpilation)"
    )
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(
        handles=[
            mpatches.Patch(color=COLOR_COUNT, label="2Q gate count"),
            mpatches.Patch(color=COLOR_DEPTH, label="2Q depth"),
        ],
        loc="upper left", fontsize=9,
    )

    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)

    out = os.path.join(RESULTS_PATH, "summit_input_2q.png")
    n = 2
    while os.path.exists(out):
        out = os.path.join(RESULTS_PATH, f"summit_input_2q_{n}.png")
        n += 1

    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
