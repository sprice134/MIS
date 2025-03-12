#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
import sys
import networkx as nx
import torch
import numpy as np

from torch_geometric.utils import from_networkx
sys.path.append('../modelAttempt2_5')
from tools3 import GCNForMIS  # Make sure this is available

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

def get_model_predictions(G, model, device):
    """
    Given a NetworkX graph G, convert it to a PyTorch Geometric Data object,
    run the model to obtain a predicted probability per node, and return a dictionary
    mapping node to predicted probability.
    """
    data = from_networkx(G)
    # Ensure the number of nodes is set correctly.
    data.num_nodes = G.number_of_nodes()
    # If no node features exist, assign a default feature of 1 for each node.
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(data)
    pred_np = pred.cpu().numpy().flatten()
    # Build dictionary: node -> predicted probability.
    predictions = {i: float(pred_np[i]) for i in range(len(pred_np))}
    return predictions

def noisy_oracle(v, predictions, epsilon, thresh):
    """
    Simulate a non-persistent noisy oracle using model predictions.
    Instead of checking for membership in a set, we compare the predicted probability for vertex v
    against the user-provided threshold (e.g. --thresh 0.2). If predictions[v] >= thresh,
    we treat v as if it is in the MIS (and return 1 with probability 0.5+epsilon);
    otherwise, we treat it as not in the MIS (and return 1 with probability 0.5-epsilon).
    Each query is independent.
    """
    if predictions[v] >= thresh:
        return 1 if random.random() < (0.5 + epsilon) else 0
    else:
        return 1 if random.random() < (0.5 - epsilon) else 0

def greedy_MIS(G_sub):
    """
    Compute an independent set using a greedy algorithm on graph G_sub.
    This algorithm repeatedly selects a vertex (here, the one with minimum degree),
    adds it to the independent set, and removes it along with its neighbors.
    """
    I = set()
    remaining = set(G_sub.nodes())
    while remaining:
        # Select the vertex with minimum degree in the remaining subgraph.
        v = min(remaining, key=lambda x: G_sub.degree(x))
        I.add(v)
        # Remove v and all its neighbors from the remaining set.
        neighbors = set(G_sub.neighbors(v))
        remaining.remove(v)
        remaining -= neighbors
    return I

def non_persistent_MIS(G, predictions, epsilon, delta, thresh):
    """
    Implements Algorithm 2 (non-persistent noise setting) using model-based predictions.
    
    Parameters:
      G           : The input graph (networkx graph).
      predictions : A dictionary mapping each node to its predicted probability.
      epsilon     : The oracle accuracy parameter (0 < epsilon < 0.5).
      delta       : Confidence parameter (0 < delta < 1).
      thresh      : Threshold for binarizing the predicted probability.
    
    Returns:
      best_independent_set: The computed independent set (as a set of vertices).
      total_queries       : Total number of oracle queries used.
      rounds              : Number of rounds performed.
    """
    n = G.number_of_nodes()
    V_r = set(G.nodes())  # initial surviving set
    best_independent_set = set()
    total_queries = 0
    Q_threshold = (30 * n / (epsilon ** 2)) * math.log(1 / delta)
    r = 1

    # Print header for round output.
    print("Round | q_r  | Surviving vertices | Greedy MIS | Independent Set | Total Queries")
    print("----------------------------------------------------------------------")
    while total_queries < Q_threshold and V_r:
        q_r = math.ceil((4 / (epsilon ** 2)) * (r + math.log(1 / delta)))
        new_V = set()
        for v in V_r:
            yes_count = 0
            for _ in range(q_r):
                yes_count += noisy_oracle(v, predictions, epsilon, thresh)
            total_queries += q_r
            if yes_count >= q_r / 2:
                new_V.add(v)
        V_r = new_V
        
        if not V_r:
            print(f"Round {r}: No surviving vertices remain.")
            break
        
        H = G.subgraph(V_r)
        # Instead of computing a vertex cover, we compute a greedy independent set.
        I_r = greedy_MIS(H)
        
        print(f"{r:5d} | {q_r:4d} | {len(V_r):19d} | {len(I_r):10d} | {len(I_r):16d} | {total_queries:13d}")
        
        if len(I_r) > len(best_independent_set):
            best_independent_set = I_r
        
        r += 1

    return best_independent_set, total_queries, r - 1

def main():
    parser = argparse.ArgumentParser(
        description="Non-persistent Noise MIS algorithm (Algorithm 2) using model predictions."
    )
    parser.add_argument("--graph_file", type=str, required=True,
                        help="Path to the edgelist file for the graph.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file with MIS info.")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Epsilon value for the oracle (0 < epsilon < 0.5).")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Confidence parameter delta (0 < delta < 1).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Number of hidden channels in the model (should match training).")
    parser.add_argument("--num_layers", type=int, default=7,
                        help="Number of layers in the model (should match training).")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Threshold for binarizing predictions.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Error: Model path must be provided.")
        sys.exit(1)
    model.eval()

    G = load_graph(args.graph_file)
    json_entry = load_json(args.json_file)
    print(f"Loaded graph with {G.number_of_nodes()} vertices.")

    expected_n = get_expected_num_nodes(args.json_file)
    if expected_n is not None:
        for i in range(expected_n):
            if i not in G.nodes():
                G.add_node(i)
        print(f"After adding isolated nodes, graph has {G.number_of_nodes()} vertices (expected {expected_n}).")
    else:
        print("Could not determine expected number of nodes from JSON file name.")

    nodes = sorted(G.nodes())
    mapping = {node: idx for idx, node in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)
    print("Graph nodes relabeled to contiguous 0-indexed order.")
    print(f"Graph now has {G.number_of_nodes()} vertices with labels: {sorted(G.nodes())}")

    predictions = get_model_predictions(G, model, device)
    binary_preds = {v: 1 if predictions[v] >= args.thresh else 0 for v in predictions}
    print(f"\nModel predicted status for each vertex (after thresholding at {args.thresh}):")
    for v in sorted(binary_preds):
        print(f"  Vertex {v}: {binary_preds[v]} (predicted probability = {predictions[v]:.3f})")

    print("\nRunning Non-persistent Noise MIS Algorithm (Algorithm 2) using model predictions...\n")
    best_I, total_queries, rounds = non_persistent_MIS(G, predictions, args.epsilon, args.delta, args.thresh)
    print(f"\nCompleted in {rounds} rounds using a total of {total_queries} oracle queries.")
    print("Computed independent set (vertex labels):")
    print(" ", sorted(best_I))

if __name__ == "__main__":
    main()

'''
python algo2_model_attempt2_single.py \
    --graph_file ../modelAttempt2_5/test_generated_graphs/nodes_80/removal_80percent/graph_14.edgelist \
    --json_file ../modelAttempt2_5/test_mis_results_grouped_v3/nodes_80_removal_80percent.json \
    --epsilon 0.1 --delta 0.1 \
    --model_path ../modelAttempt2_5/best_model_binary_32_176_28_0.001_v1.pth \
    --hidden_channels 176 \
    --num_layers 28 \
    --thresh 0.2
'''
