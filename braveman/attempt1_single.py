#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
import networkx as nx

def load_graph(graph_file):
    # Reads a graph from an edgelist file.
    G = nx.read_edgelist(graph_file, nodetype=int)
    return G

def load_json(json_file):
    # Load the JSON file and return the first entry.
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not data:
        raise ValueError("JSON file is empty!")
    return data[0]

def get_expected_num_nodes(json_file):
    # Parse the expected number of nodes from the json_file name.
    match = re.search(r"nodes_(\d+)_", json_file)
    if match:
        return int(match.group(1))
    else:
        return None

def bravemenV1(mis_set, num_cells, epsilon):
    """
    Returns an oracle vector of length num_cells.
    For each cell, if its index is in mis_set it starts as 1, otherwise 0.
    Then, it is flipped with probability (0.5 - epsilon),
    so it remains unchanged with probability (0.5+epsilon).
    """
    oracleVector = [1 if i in mis_set else 0 for i in range(num_cells)]
    for i in range(num_cells):
        if random.random() < (0.5 - epsilon):
            oracleVector[i] = 1 - oracleVector[i]
    return oracleVector

def greedy_mis_set(H):
    """
    Greedily computes an independent set on graph H.
    It repeatedly selects the vertex with minimum degree (tie-broken by vertex ID)
    and removes that vertex and its neighbors.
    """
    independent_set = set()
    H_copy = H.copy()
    while H_copy.number_of_nodes() > 0:
        node = min(H_copy.nodes(), key=lambda n: (H_copy.degree(n), n))
        independent_set.add(node)
        neighbors = list(H_copy.neighbors(node))
        H_copy.remove_node(node)
        H_copy.remove_nodes_from(neighbors)
    return independent_set

def main():
    parser = argparse.ArgumentParser(
        description="Debug persistent-noise MIS on a single graph and JSON file."
    )
    parser.add_argument("--graph_file", type=str, required=True,
                        help="Path to the edgelist file for the graph.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file with MIS info.")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Epsilon value for the oracle (0 < epsilon < 0.5).")
    args = parser.parse_args()

    # Load graph and JSON.
    G = load_graph(args.graph_file)
    json_entry = load_json(args.json_file)
    
    # Use the first MIS_SETS index as the ground-truth MIS.
    if "MIS_SETS" in json_entry and json_entry["MIS_SETS"]:
        mis_star = set(json_entry["MIS_SETS"][0])
    else:
        print("Error: No MIS_SETS found in the JSON file.")
        return

    print(f"Loaded graph with {G.number_of_nodes()} vertices.")
    print("\n1) Ground truth status (GT) for each vertex (original labels):")
    gt_status_orig = {v: 1 if v in mis_star else 0 for v in G.nodes()}
    for v in sorted(gt_status_orig):
        print(f"  Vertex {v}: {gt_status_orig[v]}")

    # Add isolated nodes if necessary.
    expected_n = get_expected_num_nodes(args.json_file)
    if expected_n is not None:
        for i in range(expected_n):
            if i not in G.nodes():
                G.add_node(i)
        print(f"\nAfter adding isolated nodes, graph has {G.number_of_nodes()} vertices (expected {expected_n}).")
    else:
        print("\nCould not determine expected number of nodes from JSON file name.")

    # Always relabel the graph to contiguous 0-indexed labels.
    nodes = sorted(G.nodes())
    mapping = {node: idx for idx, node in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)
    # Update mis_star using the same mapping.
    mis_star = {mapping[node] for node in mis_star if node in mapping}
    print("\nGraph nodes relabeled to contiguous 0-indexed order.")
    print(f"Graph now has {G.number_of_nodes()} vertices with labels: {sorted(G.nodes())}")
    
    n = G.number_of_nodes()
    # Recompute ground truth for new labels.
    gt_status = {v: 1 if v in mis_star else 0 for v in G.nodes()}

    # Step 2: Apply the noisy oracle.
    oracle_vector = bravemenV1(mis_star, n, args.epsilon)
    print("\n2) After augmentation (oracle vector):")
    for i, val in enumerate(oracle_vector):
        changed = " (changed)" if val != gt_status[i] else ""
        print(f"  Vertex {i}: {val}{changed}")

    # Step 3: Compute noisy degree for each vertex.
    deg_gI = {}
    for v in G.nodes():
        deg_gI[v] = sum(oracle_vector[w] for w in G.neighbors(v))
    print("\n3) Computed noisy degree $\\widetilde{{deg}}_{{I^*}}(v)$ for each vertex:")
    for v in sorted(deg_gI):
        print(f"  Vertex {v}: {deg_gI[v]}")

    # Step 4: Define set L = { v in V : deg(v) <= 36 ln(n) }.
    threshold_L = 36 * math.log(n)
    L = {v for v in G.nodes() if G.degree(v) <= threshold_L}
    print(f"\n4) Set L (vertices with deg(v) <= 36 ln(n)); constant threshold = {threshold_L:.4f}:")
    print("  ", sorted(L))

    # Step 5: For vertices not in L, compute threshold s_v and let S = { v in V\\L : noisy degree <= s_v }.
    S = set()
    s_values = {}
    for v in G.nodes():
        if v not in L:
            d = G.degree(v)
            s_v = (0.5 - args.epsilon) * d + 6 * math.sqrt(math.log(n)) * (0.5 - args.epsilon) * math.sqrt(d)
            s_values[v] = s_v
            if deg_gI[v] <= s_v:
                S.add(v)
    print("\n5) For vertices not in L, computed thresholds and noisy degrees:")
    for v in sorted(s_values):
        print(f"  Vertex {v}: threshold $s_v$ = {s_values[v]:.4f}, $\\widetilde{{deg}}_{{I^*}}(v)$ = {deg_gI[v]}")
    print("\nSet S (vertices in V\\L with $\\widetilde{{deg}}_{{I^*}}(v)$ <= $s_v$):")
    print("  ", sorted(S))

    # Step 6: Compute greedy MIS on the induced subgraph G[S ∪ L].
    induced_set = S.union(L)
    H_induced = G.subgraph(induced_set).copy()
    mis_result = greedy_mis_set(H_induced)
    print("\n6) Greedy MIS on $G[S \\cup L]$:")
    print("  ", sorted(mis_result))

if __name__ == "__main__":
    main()



    '''
    python attempt1_single.py \
        --graph_file ../modelAttempt2_5/test_generated_graphs/nodes_15/removal_85percent/graph_21.edgelist \
        --json_file ../modelAttempt2_5/test_mis_results_grouped_v3/nodes_15_removal_85percent.json \
        --epsilon 0.1

    '''