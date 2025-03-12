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

def main():
    parser = argparse.ArgumentParser(
        description="Greedy MIS on full graph and on graph after removing model rejected nodes."
    )
    parser.add_argument("--graph_file", type=str, required=True,
                        help="Path to the edgelist file for the graph.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file with MIS info.")
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

    # 1. Greedy MIS on the full graph.
    full_greedy = greedy_MIS(G)
    print(f"\nGreedy MIS on full graph: {len(full_greedy)}")
    print(" ", sorted(full_greedy))

    # 2. Greedy MIS on the graph after removing model rejected nodes.
    # Here, we remove nodes with predicted probability below the threshold.
    accepted_nodes = [v for v in G.nodes() if predictions[v] >= args.thresh]
    G_accepted = G.subgraph(accepted_nodes)
    accepted_greedy = greedy_MIS(G_accepted)
    print(f"\nGreedy MIS on graph after removing model rejected nodes (prediction < thresh): {len(accepted_greedy)}")
    print(" ", sorted(accepted_greedy))

if __name__ == "__main__":
    main()

'''
python algo2_greedy_single.py \
    --graph_file ../modelAttempt2_5/test_generated_graphs/nodes_80/removal_80percent/graph_14.edgelist \
    --json_file ../modelAttempt2_5/test_mis_results_grouped_v3/nodes_80_removal_80percent.json \
    --model_path ../modelAttempt2_5/best_model_binary_32_176_28_0.001_v1.pth \
    --hidden_channels 176 \
    --num_layers 28 \
    --thresh 0.2
'''
