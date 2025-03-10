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

# Assume that GCNForMIS is defined elsewhere and accessible.
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
    # Convert NetworkX graph to PyG data. (This assumes the nodes are labeled 0,1,...,n-1.)
    data = from_networkx(G)
    # Make sure the number of nodes is correctly set.
    data.num_nodes = G.number_of_nodes()
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(data)
    pred_np = pred.cpu().numpy().flatten()
    # Build dictionary: node -> predicted probability.
    predictions = {i: float(pred_np[i]) for i in range(len(pred_np))}
    return predictions

def noisy_oracle(v, predictions, epsilon):
    """
    Simulate a non-persistent noisy oracle using model predictions.
    Instead of using the ground truth, we use the model prediction for vertex v.
    The prediction is thresholded at 0.5 (i.e., >= 0.5 is taken as 1, else 0).
    Then, if the predicted label is 1, the oracle returns 1 ("yes")
    with probability 0.5+epsilon; otherwise, it returns 1 with probability 0.5-epsilon.
    Each query is independent.
    """
    pred_label = 1 if predictions[v] >= 0.5 else 0
    if pred_label == 1:
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

def non_persistent_MIS(G, predictions, epsilon, delta):
    """
    Implements Algorithm 2 (non-persistent noise setting) using model-based predictions.
    
    Parameters:
      G           : The input graph (networkx graph).
      predictions : A dictionary mapping each node to its predicted probability.
      epsilon     : The oracle accuracy parameter (0 < epsilon < 0.5).
      delta       : Confidence parameter (0 < delta < 1).
    
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

    # Continue elimination rounds until query budget is exceeded or no vertices survive.
    while total_queries < Q_threshold and V_r:
        # Set number of queries per vertex in round r.
        q_r = math.ceil((4 / (epsilon ** 2)) * (r + math.log(1 / delta)))
        
        new_V = set()
        # Elimination phase: For each vertex in current set, perform q_r queries.
        for v in V_r:
            yes_count = 0
            for _ in range(q_r):
                yes_count += noisy_oracle(v, predictions, epsilon)
            total_queries += q_r
            # Vertex survives if it receives at least half "yes" responses.
            if yes_count >= q_r / 2:
                new_V.add(v)
        V_r = new_V
        
        if not V_r:
            break
        
        # Vertex Cover phase: compute a 2-approximate vertex cover on the induced subgraph.
        H = G.subgraph(V_r)
        cover = compute_vertex_cover(H)
        I_r = V_r - cover  # The remaining vertices form an independent set.
        
        # Maintain the best (largest) independent set found so far.
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
    args = parser.parse_args()

    # Set up device and load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    if args.model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    elif args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Error: Model path must be provided.")
        sys.exit(1)
    model.eval()

    # Load graph and JSON MIS information (JSON is no longer used for oracle but only for display).
    G = load_graph(args.graph_file)
    json_entry = load_json(args.json_file)
    # Here we ignore the ground truth MIS from the JSON.
    # Instead, we use model inference predictions.

    print(f"Loaded graph with {G.number_of_nodes()} vertices.")

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
    print("Graph nodes relabeled to contiguous 0-indexed order.")
    print(f"Graph now has {G.number_of_nodes()} vertices with labels: {sorted(G.nodes())}")

    # Obtain model predictions for each node.
    predictions = get_model_predictions(G, model, device)
    # Also, display the binary prediction (thresholded at 0.5) for each vertex.
    binary_preds = {v: 1 if predictions[v] >= 0.5 else 0 for v in predictions}
    print("\nModel predicted status for each vertex (after thresholding at 0.5):")
    for v in sorted(binary_preds):
        print(f"  Vertex {v}: {binary_preds[v]} (predicted probability = {predictions[v]:.3f})")

    print("\nRunning Non-persistent Noise MIS Algorithm (Algorithm 2) using model predictions...")
    best_I, total_queries, rounds = non_persistent_MIS(G, predictions, args.epsilon, args.delta)
    print(f"\nCompleted in {rounds} rounds using a total of {total_queries} oracle queries.")
    print("Computed independent set (vertex labels):")
    print(" ", sorted(best_I))

if __name__ == "__main__":
    main()
    '''
    python algo2_model_attempt1_single.py \
        --graph_file ../modelAttempt2_5/test_generated_graphs/nodes_80/removal_80percent/graph_6.edgelist \
        --json_file ../modelAttempt2_5/test_mis_results_grouped_v3/nodes_80_removal_80percent.json \
        --epsilon 0.25 --delta 0.75 \
        --model_path ../modelAttempt2_5/best_model_prob_32_176_28_0.001_v6.pth \
        --hidden_channels 176 \
        --num_layers 28
    '''