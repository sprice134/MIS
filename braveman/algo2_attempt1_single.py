#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
import sys
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

def noisy_oracle(v, mis_set, epsilon):
    """
    Simulate a non-persistent noisy oracle.
    If vertex v is in the ground truth MIS (mis_set), the oracle returns 1 ("yes")
    with probability 0.5+epsilon; otherwise, it returns 1 with probability 0.5-epsilon.
    Each query is independent.
    """
    if v in mis_set:
        return 1 if random.random() < (0.5 + epsilon) else 0
    else:
        return 1 if random.random() < (0.5 - epsilon) else 0

def compute_vertex_cover(G_sub):
    """
    Compute a 2-approximate vertex cover of the subgraph G_sub using a maximal matching.
    Every edge in the matching contributes both endpoints.
    """
    matching = nx.algorithms.matching.maximal_matching(G_sub)
    cover = set()
    for u, v in matching:
        cover.add(u)
        cover.add(v)
    return cover

def non_persistent_MIS(G, mis_set, epsilon, delta):
    """
    Implements Algorithm 2 (non-persistent noise setting) as described in the manuscript.
    
    Parameters:
      G      : The input graph (networkx graph).
      mis_set: The ground-truth maximum independent set (for simulating the oracle).
      epsilon: The oracle accuracy parameter (0 < epsilon < 0.5).
      delta  : Confidence parameter (0 < delta < 1).
    
    Returns:
      best_independent_set: The computed independent set (as a set of vertices).
      total_queries       : Total number of oracle queries used.
      rounds              : Number of rounds performed.
    """
    n = G.number_of_nodes()
    V_r = set(G.nodes())  # initial surviving set
    best_independent_set = set()
    total_queries = 0
    # Total query threshold as given in the paper: 30*n/epsilon^2 * log(1/delta)
    Q_threshold = (30 * n / (epsilon ** 2)) * math.log(1 / delta)
    r = 1

    # Print header for round output.
    print("Round | q_r  | Surviving vertices | Vertex Cover | Independent Set | Total Queries")
    print("----------------------------------------------------------------------")
    
    # Continue elimination rounds until query budget is exceeded or no vertices survive.
    while total_queries < Q_threshold and V_r:
        # Set number of queries per vertex in round r.
        q_r = math.ceil((4 / (epsilon ** 2)) * (r + math.log(1 / delta)))
        
        new_V = set()
        # Elimination phase: For each vertex in current set, perform q_r queries.
        for v in V_r:
            yes_count = 0
            for _ in range(q_r):
                yes_count += noisy_oracle(v, mis_set, epsilon)
            total_queries += q_r
            # Vertex survives if it receives at least half "yes" responses.
            if yes_count >= q_r / 2:
                new_V.add(v)
        V_r = new_V
        
        if not V_r:
            print(f"Round {r}: No surviving vertices remain.")
            break
        
        # Vertex Cover phase: compute a 2-approximate vertex cover on the induced subgraph.
        H = G.subgraph(V_r)
        cover = compute_vertex_cover(H)
        I_r = V_r - cover  # The remaining vertices form an independent set.
        
        # Print round summary.
        print(f"{r:5d} | {q_r:4d} | {len(V_r):19d} | {len(cover):12d} | {len(I_r):16d} | {total_queries:13d}")
        
        # Maintain the best (largest) independent set found so far.
        if len(I_r) > len(best_independent_set):
            best_independent_set = I_r
        
        r += 1

    return best_independent_set, total_queries, r - 1

def main():
    parser = argparse.ArgumentParser(
        description="Non-persistent Noise MIS algorithm (Algorithm 2) from the manuscript."
    )
    parser.add_argument("--graph_file", type=str, required=True,
                        help="Path to the edgelist file for the graph.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file with MIS info.")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Epsilon value for the oracle (0 < epsilon < 0.5).")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Confidence parameter delta (0 < delta < 1).")
    args = parser.parse_args()

    # Load graph and JSON MIS information.
    G = load_graph(args.graph_file)
    json_entry = load_json(args.json_file)
    if "MIS_SETS" in json_entry and json_entry["MIS_SETS"]:
        mis_star = set(json_entry["MIS_SETS"][0])
    else:
        print("Error: No MIS_SETS found in the JSON file.")
        sys.exit(1)

    print(f"Loaded graph with {G.number_of_nodes()} vertices.")
    print("Ground truth status for each vertex (original labels):")
    gt_status_orig = {v: 1 if v in mis_star else 0 for v in G.nodes()}
    for v in sorted(gt_status_orig):
        print(f"  Vertex {v}: {gt_status_orig[v]}")

    # Add isolated nodes if the expected number is provided.
    expected_n = get_expected_num_nodes(args.json_file)
    if expected_n is not None:
        for i in range(expected_n):
            if i not in G.nodes():
                G.add_node(i)
        print(f"After adding isolated nodes, graph has {G.number_of_nodes()} vertices (expected {expected_n}).")
    else:
        print("Could not determine expected number of nodes from JSON file name.")

    # Relabel the graph to contiguous 0-indexed labels.
    nodes = sorted(G.nodes())
    mapping = {node: idx for idx, node in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)
    mis_star = {mapping[node] for node in mis_star if node in mapping}
    print("Graph nodes relabeled to contiguous 0-indexed order.")
    print(f"Graph now has {G.number_of_nodes()} vertices with labels: {sorted(G.nodes())}")

    print("\nRunning Non-persistent Noise MIS Algorithm (Algorithm 2)...")
    best_I, total_queries, rounds = non_persistent_MIS(G, mis_star, args.epsilon, args.delta)
    print(f"\nCompleted in {rounds} rounds using a total of {total_queries} oracle queries.")
    print("Computed independent set (vertex labels):")
    print(" ", sorted(best_I))

if __name__ == "__main__":
    main()


    '''
    
    python algo2_attempt1_single.py \
        --graph_file ../modelAttempt2_5/test_generated_graphs/nodes_80/removal_80percent/graph_6.edgelist \
        --json_file ../modelAttempt2_5/test_mis_results_grouped_v3/nodes_80_removal_80percent.json \
        --epsilon 0.1 --delta 0.1 

    python algo2_attempt1_single.py \
        --graph_file ../modelAttempt2_5/test_generated_graphs/nodes_80/removal_80percent/graph_6.edgelist \
        --json_file ../modelAttempt2_5/test_mis_results_grouped_v3/nodes_80_removal_80percent.json \
        --epsilon 0.25 --delta 0.75
    '''