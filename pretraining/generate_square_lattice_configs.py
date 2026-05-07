import json
import os
import pickle
import networkx as nx
from collections import defaultdict
from networkx.algorithms.isomorphism import is_isomorphic
from twisterl.utils import pull_hub_algorithm


def max_depth_and_diff_max(n):
    if n <= 5:
        return 128, 256
    elif n <= 7:
        return 256, 512
    else:
        return 512, 1024


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
    # cycles == 0 (tree)
    if max_deg == 4:
        return "X"
    if max_deg <= 2:
        return "L"
    # max_deg == 3
    deg3_nodes = [v for v, d in sg.degree() if d == 3]
    if len(deg3_nodes) >= 3:
        return "E"
    if len(deg3_nodes) == 2:
        return "H"
    # exactly one degree-3 node
    center = deg3_nodes[0]
    lengths = sorted(branch_length(sg, center, nb) for nb in sg.neighbors(center))
    l1, l2, l3 = lengths
    if l2 == l3:
        return "Y"
    if l1 == l2:
        return "T"
    return "F"


def make_config(sg, name):
    n = sg.number_of_nodes()
    max_depth, diff_max = max_depth_and_diff_max(n)
    gateset = []
    for a, b in sg.edges():
        gateset.append(["CX", [a, b]])
        gateset.append(["CX", [b, a]])
    return {
        "env_cls": "qiskit_gym.envs.synthesis.LinearFunctionEnv",
        "env": {
            "num_qubits": n,
            "difficulty": 1,
            "gateset": gateset,
            "depth_slope": 2,
            "max_depth": max_depth,
            "metrics_weights": {
                "n_cnots": 0.01,
                "n_layers_cnots": 0.01,
                "n_layers": 0.01,
                "n_gates": 0.01
            }
        },
        "policy_cls": "twisterl.nn.BasicPolicy",
        "policy": {
            "embedding_size": 512,
            "common_layers": [256],
            "policy_layers": [],
            "value_layers": []
        },
        "algorithm_cls": "twisterl.rl.PPO",
        "algorithm": {
            "collecting": {
                "num_cores": 32,
                "num_episodes": 1024,
                "lambda": 0.995,
                "gamma": 0.995
            },
            "training": {
                "num_epochs": 10,
                "vf_coef": 0.8,
                "ent_coef": 0.01,
                "clip_ratio": 0.1,
                "normalize_advantage": False
            },
            "learning": {
                "diff_threshold": 0.85,
                "diff_max": diff_max,
                "diff_metric": "ppo_deterministic"
            },
            "optimizer": {
                "lr": 0.0003
            },
            "evals": {
                "ppo_deterministic": {
                    "num_episodes": 100,
                    "deterministic": True,
                    "num_searches": 1,
                    "num_mcts_searches": 0,
                    "num_cores": 32,
                    "C": 1.41
                },
                "ppo_10": {
                    "num_episodes": 100,
                    "deterministic": False,
                    "num_searches": 10,
                    "num_mcts_searches": 0,
                    "num_cores": 32,
                    "C": 1.41
                }
            },
            "logging": {
                "log_freq": 1,
                "checkpoint_freq": 10
            }
        }
    }


def main():
    with open("unique_subgraphs.pkl", "rb") as f:
        unique_subgraphs = pickle.load(f)

    # Relabel to 0..n-1
    unique_subgraphs = {
        k: [nx.convert_node_labels_to_integers(sg) for sg in sgs]
        for k, sgs in unique_subgraphs.items()
    }

    # Load heavy hex graphs to find which subgraphs are already covered
    local_path = pull_hub_algorithm(
        repo_id="Qiskit/ai-transpiler_linear-functions",
        model_path="./models",
        revision="main",
        validate=False
    )
    heavy_hex_models = [
        "linear_function_2qL", "linear_function_3qL",
        "linear_function_4qL", "linear_function_4qY",
        "linear_function_5qL", "linear_function_5qT",
        "linear_function_6qL", "linear_function_6qT", "linear_function_6qY",
        "linear_function_7qF", "linear_function_7qH", "linear_function_7qL", "linear_function_7qT", "linear_function_7qY",
        "linear_function_8qF", "linear_function_8qJ", "linear_function_8qL", "linear_function_8qT1", "linear_function_8qT2", "linear_function_8qY",
        "linear_function_9qF1", "linear_function_9qF2", "linear_function_9qH1", "linear_function_9qH2", "linear_function_9qH3",
        "linear_function_9qJ", "linear_function_9qL", "linear_function_9qT1", "linear_function_9qT2", "linear_function_9qY",
        "linear_function_10qL",
    ]
    heavy_hex_graphs = []
    for model_name in heavy_hex_models:
        with open(f"{local_path}/{model_name}.json") as f:
            config = json.load(f)
        edges = {tuple(sorted(qubits)) for _, qubits in config["env"]["gateset"]}
        G = nx.Graph()
        G.add_edges_from(edges)
        heavy_hex_graphs.append(G)

    # Only keep subgraphs NOT covered by any heavy hex graph
    missing_subgraphs = {
        k: [sg for sg in sgs if not any(is_isomorphic(sg, hg) for hg in heavy_hex_graphs)]
        for k, sgs in unique_subgraphs.items()
    }

    # Deduplicate within missing subgraphs — keep only one representative per isomorphism class
    seen_graphs = []
    deduplicated = {}
    for k, sgs in sorted(missing_subgraphs.items()):
        unique = []
        for sg in sgs:
            if not any(sg.number_of_nodes() == s.number_of_nodes() and is_isomorphic(sg, s) for s in seen_graphs):
                seen_graphs.append(sg)
                unique.append(sg)
        if unique:
            deduplicated[k] = unique
    missing_subgraphs = deduplicated

    total_missing = sum(len(v) for v in missing_subgraphs.values())
    print(f"Missing subgraphs to generate: {total_missing}")

    output_dir = "square_lattice_configs"
    os.makedirs(output_dir, exist_ok=True)

    # Group by (n, topology) to handle numbering when multiple same type
    groups = defaultdict(list)
    for k, sgs in sorted(missing_subgraphs.items()):
        for sg in sgs:
            letter = topology_letter(sg)
            groups[(k, letter)].append(sg)

    for (n, letter), sgs in sorted(groups.items()):
        for i, sg in enumerate(sgs):
            if len(sgs) == 1:
                name = f"linear_function_{n}q{letter}"
            else:
                name = f"linear_function_{n}q{letter}{i + 1}"
            config = make_config(sg, name)
            path = os.path.join(output_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  {name}.json  edges={list(sg.edges())}")

    print(f"\nGenerated {sum(len(v) for v in groups.values())} configs in '{output_dir}/'")


if __name__ == "__main__":
    main()
