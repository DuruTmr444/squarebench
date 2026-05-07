"""Generate unique_subgraphs.pkl — all connected induced subgraphs with 2–8 nodes
in a 5×5 square lattice, deduplicated up to isomorphism."""

import pickle
from itertools import combinations

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic


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


def main():
    # Step 1: create 5×5 square lattice
    G = nx.grid_2d_graph(5, 5)
    G = nx.convert_node_labels_to_integers(G)
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # Step 2: enumerate connected induced subgraphs
    subgraphs = induced_connected_subgraphs(G, 2, 8)
    for k, sgs in subgraphs.items():
        print(f"Connected subgraphs with {k} nodes: {len(sgs)}")

    # Step 3: deduplicate via isomorphism and relabel to 0..n-1
    unique_subgraphs = {}
    for k, sgs in subgraphs.items():
        unique = unique_up_to_isomorphism(sgs)
        unique_subgraphs[k] = [nx.convert_node_labels_to_integers(sg) for sg in unique]
        print(f"Unique connected subgraphs with {k} nodes: {len(unique)}")

    with open("unique_subgraphs.pkl", "wb") as f:
        pickle.dump(unique_subgraphs, f)
    print("Saved unique_subgraphs.pkl")


if __name__ == "__main__":
    main()
