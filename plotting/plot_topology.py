"""Plot the qubit coupling topology from a square lattice config JSON."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_coupling_map(config_path: str) -> tuple[int, list[list[int]]]:
    with open(config_path) as f:
        config = json.load(f)

    num_qubits = config["env"]["num_qubits"]
    gateset = config["env"]["gateset"]

    # Extract directed CX edges: [control, target]
    edges = [qubits for gate, qubits in gateset if gate == "CX"]
    return num_qubits, edges


def compute_qubit_coordinates(
    num_qubits: int, edges: list[list[int]]
) -> list[list[int]]:
    """Use networkx spring layout to infer 2D qubit positions."""
    G = nx.Graph()
    G.add_nodes_from(range(num_qubits))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Scale to [0, 10] grid and convert to [row, col]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    scale = 5
    coords = []
    for q in range(num_qubits):
        x, y = pos[q]
        col = round((x - x_min) / x_range * scale)
        row = round((y_max - y) / y_range * scale)  # flip y so row 0 is top
        coords.append([row, col])

    return coords


def plot_topology(config_path: str, output_path: str | None = None) -> None:
    num_qubits, edges = load_coupling_map(config_path)
    coords = compute_qubit_coordinates(num_qubits, edges)

    from qiskit.visualization import plot_coupling_map

    fig = plot_coupling_map(
        num_qubits=num_qubits,
        qubit_coordinates=coords,
        coupling_map=edges,
        plot_directed=True,
        label_qubits=True,
    )

    title = Path(config_path).stem
    fig.suptitle(title, fontsize=12)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot coupling topology from a square lattice config JSON."
    )
    parser.add_argument("config", help="Path to the config JSON file")
    parser.add_argument("-o", "--output", help="Save plot to file instead of displaying")
    args = parser.parse_args()

    plot_topology(args.config, args.output)
