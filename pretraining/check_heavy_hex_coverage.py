"""Check which connected subgraphs (2–8 nodes) of the heavy hex lattice are covered
by models in Qiskit/ai-transpiler_linear-functions.

Uses heavy_hex_graph(d=3) as the reference graph (19 nodes, 20 edges).
Since the shortest cycle in heavy hex is length 10, all subgraphs with ≤8 nodes are trees.

Usage:
    python pretraining/check_heavy_hex_coverage.py
    python pretraining/check_heavy_hex_coverage.py --target-hf Qiskit/ai-transpiler_linear-functions
    python pretraining/check_heavy_hex_coverage.py --target models/Qiskit
"""

import argparse
import json
import os
import tempfile
from itertools import combinations

import networkx as nx
import rustworkx as rx
from networkx.algorithms.isomorphism import is_isomorphic


# ── topology classification (same as generate_square_lattice_configs.py) ──────

def branch_length(sg, center, neighbor):
    prev, cur = center, neighbor
    length = 1
    while sg.degree(cur) == 2:
        nxt = [n for n in sg.neighbors(cur) if n != prev][0]
        prev, cur = cur, nxt
        length += 1
    return length


def topology_letter(sg):
    n = sg.number_of_nodes()
    cycles = sg.number_of_edges() - n + 1
    degrees = sorted(d for _, d in sg.degree())
    max_deg = max(degrees)

    if cycles > 1:
        return "B"
    if cycles == 1:
        return "S"
    if max_deg == 4:
        return "X"
    if max_deg <= 2:
        return "L"
    deg3_nodes = [v for v, d in sg.degree() if d == 3]
    if len(deg3_nodes) >= 3:
        return "E"
    if len(deg3_nodes) == 2:
        return "H"
    center = deg3_nodes[0]
    lengths = sorted(branch_length(sg, center, nb) for nb in sg.neighbors(center))
    l1, l2, l3 = lengths
    if l2 == l3:
        return "Y"
    if l1 == l2:
        return "T"
    return "F"


# ── subgraph enumeration ───────────────────────────────────────────────────────

def build_heavy_hex(d=3):
    """Build heavy hex graph (d=3 → 19 nodes) as a NetworkX graph."""
    rG = rx.generators.heavy_hex_graph(d)
    G = nx.Graph()
    G.add_edges_from(rG.edge_list())
    G = nx.convert_node_labels_to_integers(G)
    return G


def induced_connected_subgraphs(G, min_k=2, max_k=8):
    nodes = list(G.nodes())
    subgraphs = {k: [] for k in range(min_k, max_k + 1)}
    for k in range(min_k, max_k + 1):
        for subset in combinations(nodes, k):
            sg = G.subgraph(subset)
            if nx.is_connected(sg):
                subgraphs[k].append(sg.copy())
    return subgraphs


def unique_up_to_isomorphism(subgraphs):
    unique = []
    for sg in subgraphs:
        if not any(is_isomorphic(sg, u) for u in unique):
            unique.append(sg)
    return unique


# ── model loading ──────────────────────────────────────────────────────────────

def load_graphs_from_dir(directory):
    graphs = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json") or fname == "config.json":
            continue
        name = fname[:-5]
        path = os.path.join(directory, fname)
        try:
            with open(path) as f:
                config = json.load(f)
            env = config.get("env") or config.get("algorithm", {})
            gateset = env.get("gateset") if isinstance(env, dict) else None
            if not gateset:
                continue
            edges = {tuple(sorted(qubits)) for _, qubits in gateset}
            G = nx.Graph()
            G.add_edges_from(edges)
            graphs[name] = G
        except Exception as e:
            print(f"  [warn] could not load {fname}: {e}")
    return graphs


def load_graphs_from_hf(repo_id, revision="main"):
    from twisterl.utils import pull_hub_algorithm
    print(f"Downloading {repo_id}@{revision} ...")
    local_path = pull_hub_algorithm(
        repo_id=repo_id,
        model_path=tempfile.mkdtemp(),
        revision=revision,
        validate=False,
    )
    return load_graphs_from_dir(local_path)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tgt = parser.add_mutually_exclusive_group(required=False)
    tgt.add_argument("--target", metavar="DIR", help="Local directory of model configs (default: Qiskit HF repo)")
    tgt.add_argument("--target-hf", metavar="REPO_ID", default="Qiskit/ai-transpiler_linear-functions",
                     help="HuggingFace repo ID to check against (default: %(default)s)")
    parser.add_argument("--target-revision", default="main", help="HF revision (default: main)")
    parser.add_argument("--max-nodes", type=int, default=8, help="Maximum subgraph size to enumerate (default: 8)")
    args = parser.parse_args()

    # Step 1: build reference heavy hex graph
    d = 3
    G = build_heavy_hex(d)
    print(f"\nReference: heavy_hex_graph(d={d})")
    print(f"  Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"  Degree sequence: {sorted(dict(G.degree()).values(), reverse=True)}")
    shortest_cycle = len(min(nx.minimum_cycle_basis(G), key=len))
    print(f"  Shortest cycle: {shortest_cycle}  (all {args.max_nodes}-node subgraphs are trees)")

    # Step 2: enumerate and deduplicate subgraphs
    print(f"\nEnumerating connected induced subgraphs (2–{args.max_nodes} nodes)...")
    raw_subgraphs = induced_connected_subgraphs(G, 2, args.max_nodes)
    unique_subgraphs = {}
    for k, sgs in raw_subgraphs.items():
        unique = unique_up_to_isomorphism(sgs)
        unique_subgraphs[k] = [nx.convert_node_labels_to_integers(sg) for sg in unique]
        print(f"  {k} nodes: {len(sgs):5d} connected  →  {len(unique):3d} unique")

    total_unique = sum(len(v) for v in unique_subgraphs.values())
    print(f"\nTotal unique heavy hex subgraphs: {total_unique}")

    # Step 3: load target models
    print("\n=== Loading target models ===")
    if args.target:
        target_graphs = load_graphs_from_dir(args.target)
    else:
        target_graphs = load_graphs_from_hf(args.target_hf, args.target_revision)
    print(f"Loaded {len(target_graphs)} model configs.")

    # Step 4: check coverage
    covered = {}
    missing = {}
    for k, sgs in unique_subgraphs.items():
        covered[k] = []
        missing[k] = []
        for sg in sgs:
            match = next(
                (name for name, tg in target_graphs.items()
                 if sg.number_of_nodes() == tg.number_of_nodes() and is_isomorphic(sg, tg)),
                None
            )
            if match:
                covered[k].append((sg, match))
            else:
                missing[k].append(sg)

    # Step 5: report
    print("\n=== Coverage by size ===")
    print(f"  {'n':>2}  {'unique':>6}  {'covered':>7}  {'missing':>7}")
    for k in sorted(unique_subgraphs):
        n_cov = len(covered[k])
        n_mis = len(missing[k])
        n_tot = n_cov + n_mis
        print(f"  {k:>2}  {n_tot:>6}  {n_cov:>7}  {n_mis:>7}")

    total_covered = sum(len(v) for v in covered.values())
    total_missing = sum(len(v) for v in missing.values())
    print(f"\n  Total  {total_unique:>6}  {total_covered:>7}  {total_missing:>7}")

    if total_covered > 0:
        print("\n=== Covered subgraphs ===")
        for k in sorted(covered):
            for sg, match in covered[k]:
                letter = topology_letter(sg)
                print(f"  {k}q{letter}  edges={list(sg.edges())}  →  {match}")

    if total_missing > 0:
        print("\n=== Missing subgraphs (not in target) ===")
        for k in sorted(missing):
            for sg in missing[k]:
                letter = topology_letter(sg)
                print(f"  {k}q{letter}  edges={list(sg.edges())}")
    else:
        print("\nAll heavy hex subgraphs are covered.")


if __name__ == "__main__":
    main()
