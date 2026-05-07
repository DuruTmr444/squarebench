"""
Check which model configs in a source location are isomorphic to models in a target location.

Source can be a local directory or a HuggingFace repo.
Target can be a local directory or a HuggingFace repo.

Usage:
    # Check local square lattice configs against Qiskit HF repo
    python evaluation/check_isomorphism.py \
        --source square_lattice_configs \ 7qH1-7qH2
        --target-hf Qiskit/ai-transpiler_linear-functions
        --check-self-duplicates 8qH4=8qH5

    # Check DuruTo HF repo against Qiskit HF repo
    python evaluation/check_isomorphism.py \
        --source-hf DuruTo/ai-transpiler_linear-functions \
        --target-hf Qiskit/ai-transpiler_linear-functions

    # Check local against local
    python evaluation/check_isomorphism.py \
        --source square_lattice_configs \
        --target models/Qiskit

    # Also check for duplicates within the source itself
    python evaluation/check_isomorphism.py \
        --source square_lattice_configs \
        --target-hf Qiskit/ai-transpiler_linear-functions \
        --check-self-duplicates
"""

import argparse
import json
import os
import tempfile

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic


def load_graphs_from_dir(directory):
    """Load all model JSON configs from a local directory. Returns {name: nx.Graph}."""
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
            print(f"  [warn] Could not load {fname}: {e}")
    return graphs


def load_graphs_from_hf(repo_id, revision="main"):
    """Download a HF repo and load all model configs. Returns {name: nx.Graph}."""
    from twisterl.utils import pull_hub_algorithm
    print(f"Downloading {repo_id}@{revision} ...")
    local_path = pull_hub_algorithm(
        repo_id=repo_id,
        model_path=tempfile.mkdtemp(),
        revision=revision,
        validate=False,
    )
    return load_graphs_from_dir(local_path)


def load_graphs(local=None, hf=None, revision="main"):
    if hf:
        return load_graphs_from_hf(hf, revision)
    if local:
        return load_graphs_from_dir(local)
    raise ValueError("Must provide either --source/--target or --source-hf/--target-hf")


def find_isomorphic_match(g, target_graphs):
    """Return the name of the first target graph isomorphic to g, or None."""
    for tname, tg in target_graphs.items():
        if g.number_of_nodes() == tg.number_of_nodes() and is_isomorphic(g, tg):
            return tname
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Check which source model configs are isomorphic to target models."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--source",    metavar="DIR",     help="Local directory of source configs")
    src.add_argument("--source-hf", metavar="REPO_ID", help="HuggingFace repo ID of source configs")

    tgt = parser.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--target",    metavar="DIR",     help="Local directory of target configs")
    tgt.add_argument("--target-hf", metavar="REPO_ID", help="HuggingFace repo ID of target configs")

    parser.add_argument("--source-revision", default="main", help="HF revision for source (default: main)")
    parser.add_argument("--target-revision", default="main", help="HF revision for target (default: main)")
    parser.add_argument("--check-self-duplicates", action="store_true",
                        help="Also report duplicate pairs within the source itself")
    args = parser.parse_args()

    print("\n=== Loading source ===")
    source_graphs = load_graphs(local=args.source, hf=args.source_hf, revision=args.source_revision)
    print(f"Loaded {len(source_graphs)} source configs.")

    print("\n=== Loading target ===")
    target_graphs = load_graphs(local=args.target, hf=args.target_hf, revision=args.target_revision)
    print(f"Loaded {len(target_graphs)} target configs.")

    # --- Check source against target ---
    print("\n=== Source configs isomorphic to a target model ===")
    matches = []
    for sname, sg in sorted(source_graphs.items()):
        match = find_isomorphic_match(sg, target_graphs)
        if match:
            matches.append((sname, match))
            print(f"  {sname}  <->  {match}")

    if not matches:
        print("  None found.")
    print(f"\nTotal matches: {len(matches)} / {len(source_graphs)}")

    # --- Check within source for duplicates ---
    if args.check_self_duplicates:
        print("\n=== Duplicate pairs within source ===")
        names = sorted(source_graphs.keys())
        dupes = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ga, gb = source_graphs[a], source_graphs[b]
                if ga.number_of_nodes() == gb.number_of_nodes() and is_isomorphic(ga, gb):
                    dupes.append((a, b))
                    print(f"  {a}  <->  {b}")
        if not dupes:
            print("  None found.")
        print(f"\nTotal duplicate pairs: {len(dupes)}")

    # --- Summary ---
    redundant = {s for s, _ in matches}
    if args.check_self_duplicates:
        for a, b in dupes:
            redundant.add(b)  # keep a, flag b as redundant
    print(f"\n=== Summary ===")
    print(f"  Source configs:          {len(source_graphs)}")
    print(f"  Isomorphic to target:    {len(matches)}")
    if args.check_self_duplicates:
        print(f"  Internal duplicates:     {len(dupes)} pairs")
    print(f"  Redundant (to remove):   {len(redundant)}")
    if redundant:
        print("  Files to remove:")
        for name in sorted(redundant):
            print(f"    {name}")


if __name__ == "__main__":
    main()
