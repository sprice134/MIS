#!/usr/bin/env python3
"""
Improved and consistent version.

This script processes MIS graphs on a per-(JSON, edgelist) pair basis.
For each JSON file (named like:
    nodes_{num_nodes}_removal_{removal_percent}percent.json
and containing a list of graph dictionaries with MIS information) and its corresponding
edgelist directory, the script:
  - Loads the MIS JSON entries and builds a mapping from edgelist basename → MIS info.
  - Lists all edgelist files (sorted) from the edgelist directory.
  - Loads all graphs from the edgelist directory (using MISGraphDataset).
  - For each graph, uses the index in the mini-dataset to pick the corresponding edgelist filename.
  - Uses that filename to look up the true MIS size from the JSON entry.
  - Extracts the numeric graph ID from the edgelist filename (e.g. "graph_10.edgelist" → 10).
  - Computes six greedy MIS approximations:
       1. Greedy MIS (minimum degree),
       2. Greedy MIS (random selection averaged over 5 runs),
       3. Greedy MIS (using JSON cell probabilities, highest probability first),
       4. Greedy MIS (selecting the node with the lowest sum of neighboring JSON probabilities),
       5. Greedy MIS (using model-predicted probabilities, highest probability first),
       6. Greedy MIS (selecting the node with the lowest sum of neighboring model probabilities).
  - Writes a CSV row per graph with: num_nodes, removal_percent, graph_id, file_graph_id, true_mis,
    mis_min_degree, mis_random, mis_prob, mis_low_neighbor_prob, mis_prob_model, mis_low_neighbor_prob_model.
    
Example:
    /home/sprice/MIS/modelAttempt2_5/generated_graphs/nodes_45/removal_30percent/graph_10.edgelist
would yield a file_graph_id of 10.
"""

import os
import csv
import json
import torch
import random
import numpy as np
import argparse

from torch_geometric.utils import to_networkx
from tools3 import MISGraphDataset, GCNForMIS  # Ensure these are accessible
import networkx as nx

# Set seeds for reproducibility.
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def gather_json_and_edgelist_paths(json_dir, node_counts, removal_percents):
    """
    For each combination of node count and removal percentage, build the expected JSON file
    path (from json_dir) and the corresponding edgelist directory path.
    Returns two lists of equal length.
    """
    json_paths = []
    edgelist_dirs = []
    
    # Assume edgelist directories are under: <parent_dir>/test_generated_graphs
    parent_dir = os.path.dirname(json_dir)
    edgelist_base = os.path.join(parent_dir, "test_generated_graphs")
    
    for n in node_counts:
        for percent in removal_percents:
            json_filename = f"nodes_{n}_removal_{percent}percent.json"
            json_path = os.path.join(json_dir, json_filename)
            if not os.path.exists(json_path):
                print(f"Warning: JSON file '{json_path}' does not exist. Skipping.")
                continue
            json_paths.append(json_path)
            
            edgelist_dir = os.path.join(edgelist_base, f"nodes_{n}", f"removal_{percent}percent")
            if not os.path.exists(edgelist_dir):
                print(f"Warning: Edgelist directory '{edgelist_dir}' does not exist. Skipping.")
                continue
            edgelist_dirs.append(edgelist_dir)
    
    return json_paths, edgelist_dirs


def build_json_mapping(json_path):
    """
    Loads the JSON file and returns a dictionary mapping the edgelist file basename
    (e.g. "graph_100.edgelist") to its MIS info.
    """
    mapping = {}
    try:
        with open(json_path, "r") as f:
            data_list = json.load(f)
        for entry in data_list:
            key = os.path.basename(entry.get("file_path", ""))
            if key:
                mapping[key] = entry
            else:
                print(f"Warning: An entry in {json_path} is missing 'file_path'.")
    except Exception as e:
        print(f"Error reading JSON from {json_path}: {e}")
    return mapping


def list_edgelist_files(edgelist_dir):
    """
    Returns a sorted list of edgelist filenames from the given directory.
    """
    try:
        files = [f for f in os.listdir(edgelist_dir) if f.endswith(".edgelist")]
        return sorted(files)
    except Exception as e:
        print(f"Error listing edgelist files in {edgelist_dir}: {e}")
        return []


def extract_graph_id(file_key):
    """
    Given an edgelist file name like "graph_10.edgelist", extract and return the numeric graph ID.
    """
    try:
        parts = file_key.split("_")
        if len(parts) >= 2:
            id_part = parts[1]
            id_str = id_part.split(".")[0]
            return int(id_str)
        else:
            print(f"Warning: Unexpected file key format: {file_key}")
            return -1
    except Exception as e:
        print(f"Error extracting graph id from {file_key}: {e}")
        return -1


def greedy_mis_min_degree(G):
    """
    Greedy algorithm: repeatedly select the node with the minimum degree.
    Uses a tie-breaker on the node ID for consistency.
    """
    H = G.copy()
    independent_set = set()
    while H.number_of_nodes() > 0:
        node = min(H.nodes(), key=lambda n: (H.degree[n], n))
        independent_set.add(node)
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return len(independent_set)


def greedy_mis_random(G):
    """
    Greedy algorithm: randomly select a node and remove it and its neighbors.
    """
    H = G.copy()
    independent_set = set()
    while list(H.nodes()):
        node = random.choice(list(H.nodes()))
        independent_set.add(node)
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return len(independent_set)


def greedy_mis_random_average(G, n_runs=5):
    sizes = [greedy_mis_random(G) for _ in range(n_runs)]
    return sum(sizes) / len(sizes)


def greedy_mis_prob(G, node_probs):
    """
    Greedy algorithm: select nodes in descending order of probability.
    Deterministic tie-breaking is enforced by sorting by (-probability, node).
    """
    sorted_nodes = sorted(node_probs.items(), key=lambda x: (-x[1], x[0]))
    independent_set = set()
    selected = set()
    for node, prob in sorted_nodes:
        if not any(neighbor in selected for neighbor in G.neighbors(node)):
            independent_set.add(node)
            selected.add(node)
    return len(independent_set)


def greedy_mis_low_neighbor_prob(G, node_probs):
    """
    Greedy algorithm: repeatedly select the node whose current neighbors have the lowest total probability.
    Iterates over sorted nodes for a deterministic tie-breaker.
    """
    H = G.copy()
    independent_set = set()
    while H.number_of_nodes() > 0:
        best_node = None
        best_neighbor_prob_sum = float('inf')
        for node in sorted(H.nodes()):
            neighbor_prob_sum = sum(node_probs[neigh] for neigh in sorted(H.neighbors(node)))
            if neighbor_prob_sum < best_neighbor_prob_sum:
                best_neighbor_prob_sum = neighbor_prob_sum
                best_node = node
        if best_node is None:
            break
        independent_set.add(best_node)
        neighbors = list(H.neighbors(best_node))
        H.remove_node(best_node)
        H.remove_nodes_from(neighbors)
    return len(independent_set)


def main():
    parser = argparse.ArgumentParser(
        description="Compute greedy MIS approximations for MIS graphs and write results to CSV."
    )
    parser.add_argument("--node_counts", type=int, nargs="+", default=list(range(10, 55, 5)),
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", default=list(range(15, 90, 5)),
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--json_dir", type=str, default="/home/sprice/MIS/modelAttempt2_5/mis_results_grouped_v3",
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--csv_out", type=str, default="mis_greedy_results.csv",
                        help="Output CSV filename.")
    # Arguments for model-based evaluation:
    parser.add_argument("--model_path", type=str, default="best_model_prob.pth",
                        help="Path to the trained model.")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Number of hidden channels in the GCN (should match training).")
    parser.add_argument("--num_layers", type=int, default=7,
                        help="Number of layers in the GCN (should match training).")
    args = parser.parse_args()

    # Load the trained model for model-based evaluations.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from '{args.model_path}'.")
    else:
        print(f"Warning: Model path '{args.model_path}' not found. Model-based evaluations will be skipped.")
        model = None
    if model is not None:
        model.eval()

    # Gather JSON file paths and corresponding edgelist directories.
    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(
        json_dir=args.json_dir,
        node_counts=args.node_counts,
        removal_percents=args.removal_percents
    )
    
    if not json_paths or not edgelist_dirs:
        print("Error: No valid JSON files or edgelist directories found. Exiting.")
        return

    print(f"Found {len(json_paths)} (JSON, edgelist) pairs.")

    # CSV header includes extra columns for model-based metrics.
    csv_header = [
        "num_nodes", "removal_percent", "graph_id", "file_graph_id", "true_mis",
        "mis_min_degree", "mis_random", "mis_prob", "mis_low_neighbor_prob",
        "mis_prob_model", "mis_low_neighbor_prob_model"
    ]
    out_rows = []
    global_graph_id = 0

    # Process each (JSON, edgelist) pair.
    for json_path, edgelist_dir in zip(json_paths, edgelist_dirs):
        json_mapping = build_json_mapping(json_path)
        sample_entry = next(iter(json_mapping.values()), {})
        percent_from_json = sample_entry.get("percent_removed", None)
        if percent_from_json is None:
            print(f"Warning: No percent_removed found in {json_path}; using value from filename.")
            percent_from_json = int(os.path.basename(json_path).split("removal_")[1].split("percent")[0])
        
        edgelist_files = list_edgelist_files(edgelist_dir)
        if not edgelist_files:
            print(f"Warning: No edgelist files found in {edgelist_dir}; skipping this pair.")
            continue

        print(f"Processing edgelist dir '{edgelist_dir}' with removal_percent={percent_from_json}.")
        
        # Load graphs from this edgelist directory (and corresponding JSON).
        dataset = MISGraphDataset(
            json_paths=[json_path],
            edgelist_dirs=[edgelist_dir],
            label_type='prob'
        )
        print(f"  Loaded {len(dataset)} graphs from {edgelist_dir}.")

        if len(dataset) != len(edgelist_files):
            print(f"Warning: Number of graphs ({len(dataset)}) does not match number of edgelist files ({len(edgelist_files)}) in {edgelist_dir}.")

        for idx, data in enumerate(dataset):
            # Convert the PyG Data object to a NetworkX graph.
            G = to_networkx(data, to_undirected=True)
            # Ensure all nodes (including isolated ones) are present.
            G.add_nodes_from(range(data.num_nodes))
            num_nodes = data.num_nodes

            # Use the index to pick a file from edgelist_files.
            file_key = edgelist_files[idx] if idx < len(edgelist_files) else edgelist_files[-1]
            file_graph_id = extract_graph_id(file_key)
            true_mis = int(json_mapping.get(file_key, {}).get("MIS_SIZE", -1))

            # Compute the six greedy MIS approximations.
            mis_min_degree = greedy_mis_min_degree(G)
            mis_random = greedy_mis_random_average(G, n_runs=5)
            y = data.y.cpu().numpy().flatten()
            node_probs = {i: float(y[i]) for i in range(len(y))}
            mis_prob = greedy_mis_prob(G, node_probs)
            mis_low_neighbor_prob = greedy_mis_low_neighbor_prob(G, node_probs)

            if model is not None:
                data_device = data.to(device)
                with torch.no_grad():
                    pred = model(data_device)
                pred_np = pred.cpu().numpy().flatten()
                node_probs_model = {i: float(pred_np[i]) for i in range(len(pred_np))}
                mis_prob_model = greedy_mis_prob(G, node_probs_model)
                mis_low_neighbor_prob_model = greedy_mis_low_neighbor_prob(G, node_probs_model)
            else:
                mis_prob_model = -1
                mis_low_neighbor_prob_model = -1

            out_rows.append([
                num_nodes,
                percent_from_json,
                global_graph_id,
                file_graph_id,
                true_mis,
                mis_min_degree,
                mis_random,
                mis_prob,
                mis_low_neighbor_prob,
                mis_prob_model,
                mis_low_neighbor_prob_model
            ])
            print(f"  Graph {global_graph_id}: nodes={num_nodes}, TRUE_MIS={true_mis}, "
                  f"min_deg={mis_min_degree}, random={mis_random:.2f}, GT_prob={mis_prob}, GT_low_neigh={mis_low_neighbor_prob}, "
                  f"model_prob={mis_prob_model}, model_low_neigh={mis_low_neighbor_prob_model}")
            global_graph_id += 1

    with open(args.csv_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        writer.writerows(out_rows)

    print(f"\nResults written to {args.csv_out}")


if __name__ == "__main__":
    main()
    '''
    python greedyEvalV3.py \
    --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95\
    --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
    --json_dir /home/sprice/MIS/modelAttempt2_5/test_mis_results_grouped_v3 \
    --csv_out mis_greedy_results_v6.csv \
    --model_path best_model_prob_32_176_28_0.001_v6.pth \
    --hidden_channels 176 \
    --num_layers 28
    '''