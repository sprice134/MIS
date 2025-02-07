#!/usr/bin/env python3
"""
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
  - Computes four greedy MIS approximations:
       1. Greedy MIS (minimum degree),
       2. Greedy MIS (random selection averaged over 5 runs),
       3. Greedy MIS (using cell probabilities from data.y, highest probability first),
       4. Greedy MIS (selecting the node with the lowest sum of neighboring probabilities).
  - Writes a CSV row per graph with: num_nodes, removal_percent, graph_id, file_graph_id,
    true_mis, mis_min_degree, mis_random, mis_prob, mis_low_neighbor_prob.
    
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
from tools3 import MISGraphDataset  # This dataset should load graphs given lists of JSON paths & edgelist directories
import networkx as nx

# Set seeds for reproducibility
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
    path (from the provided json_dir) and the corresponding edgelist directory path.
    Returns two lists of equal length.
    """
    json_paths = []
    edgelist_dirs = []
    
    # If json_dir is e.g. /home/sprice/MIS/modelAttempt2_5/mis_results_grouped_v3,
    # then the edgelist directories are assumed to be under:
    # /home/sprice/MIS/modelAttempt2_5/generated_graphs
    parent_dir = os.path.dirname(json_dir)
    edgelist_base = os.path.join(parent_dir, "generated_graphs")
    
    for n in node_counts:
        for percent in removal_percents:
            json_filename = f"nodes_{n}_removal_{percent}percent.json"
            json_path = os.path.join(json_dir, json_filename)
            if not os.path.exists(json_path):
                print(f"Warning: JSON file '{json_path}' does not exist. Skipping.")
                continue
            json_paths.append(json_path)
            
            # Build the edgelist directory path.
            edgelist_dir = os.path.join(edgelist_base, f"nodes_{n}", f"removal_{percent}percent")
            if not os.path.exists(edgelist_dir):
                print(f"Warning: Edgelist directory '{edgelist_dir}' does not exist. Skipping.")
                continue
            edgelist_dirs.append(edgelist_dir)
    
    return json_paths, edgelist_dirs


def build_json_mapping(json_path):
    """
    Loads the JSON file (which is a list of MIS entries) and returns a dictionary
    mapping the edgelist file basename (e.g. "graph_100.edgelist") to its MIS info.
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
    Returns a sorted list of edgelist filenames (basenames) from the given directory.
    """
    try:
        files = [f for f in os.listdir(edgelist_dir) if f.endswith(".edgelist")]
        return sorted(files)
    except Exception as e:
        print(f"Error listing edgelist files in {edgelist_dir}: {e}")
        return []


def extract_graph_id(file_key):
    """
    Given an edgelist file name like "graph_10.edgelist",
    extract and return the numeric graph ID (e.g. 10).
    """
    try:
        # Expecting "graph_10.edgelist"
        parts = file_key.split("_")
        if len(parts) >= 2:
            id_part = parts[1]
            # Remove the extension if present.
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
    """
    H = G.copy()
    independent_set = set()
    while H.number_of_nodes() > 0:
        node, _ = min(H.degree, key=lambda x: x[1])
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
    Greedy algorithm: select nodes in descending order of cell probability.
    """
    sorted_nodes = sorted(node_probs.items(), key=lambda x: x[1], reverse=True)
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
    For each iteration, for each node in the current graph, compute the sum of probabilities of its neighbors,
    and select the node with the minimum sum. Then, remove the selected node and all its neighbors.
    """
    H = G.copy()
    independent_set = set()
    while H.number_of_nodes() > 0:
        best_node = None
        best_neighbor_prob_sum = float('inf')
        for node in H.nodes():
            # Sum probabilities of neighbors that are still in H.
            neighbor_prob_sum = sum(node_probs[neigh] for neigh in H.neighbors(node))
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
    args = parser.parse_args()

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

    # CSV header now includes an extra column for the new metric.
    csv_header = ["num_nodes", "removal_percent", "graph_id", "file_graph_id", "true_mis",
                  "mis_min_degree", "mis_random", "mis_prob", "mis_low_neighbor_prob"]
    out_rows = []
    global_graph_id = 0

    # Process each (JSON, edgelist) pair.
    for json_path, edgelist_dir in zip(json_paths, edgelist_dirs):
        json_mapping = build_json_mapping(json_path)
        # Optionally, use the percent from one JSON entry if available.
        sample_entry = next(iter(json_mapping.values()), {})
        percent_from_json = sample_entry.get("percent_removed", None)
        if percent_from_json is None:
            print(f"Warning: No percent_removed found in {json_path}; using value from filename.")
            percent_from_json = int(os.path.basename(json_path).split("removal_")[1].split("percent")[0])
        
        # List edgelist files (basenames) from this directory.
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
            G = to_networkx(data, to_undirected=True)
            num_nodes = G.number_of_nodes()

            # Use the index in the mini-dataset to select a file from edgelist_files.
            if idx < len(edgelist_files):
                file_key = edgelist_files[idx]
            else:
                print(f"Warning: Index {idx} out of range for edgelist_files in {edgelist_dir}. Using last file name.")
                file_key = edgelist_files[-1]

            # Extract the numeric graph id from the file key.
            file_graph_id = extract_graph_id(file_key)

            if file_key in json_mapping:
                true_mis = int(json_mapping[file_key].get("MIS_SIZE", -1))
            else:
                print(f"Warning: No JSON entry found for edgelist file '{file_key}'. Setting true_mis=-1.")
                true_mis = -1

            mis_min_degree = greedy_mis_min_degree(G)
            mis_random = greedy_mis_random_average(G, n_runs=5)
            y = data.y.cpu().numpy().flatten()
            node_probs = {i: float(y[i]) for i in range(len(y))}
            mis_prob = greedy_mis_prob(G, node_probs)
            mis_low_neighbor_prob = greedy_mis_low_neighbor_prob(G, node_probs)

            out_rows.append([
                num_nodes,
                percent_from_json,
                global_graph_id,
                file_graph_id,
                true_mis,
                mis_min_degree,
                mis_random,
                mis_prob,
                mis_low_neighbor_prob
            ])
            print(f"  Processed graph {global_graph_id}: nodes={num_nodes}, file_graph_id={file_graph_id}, true_mis={true_mis}, "
                  f"min_deg={mis_min_degree}, random_avg={mis_random:.2f}, prob={mis_prob}, low_neigh_prob={mis_low_neighbor_prob}")
            global_graph_id += 1

    with open(args.csv_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        writer.writerows(out_rows)

    print(f"\nResults written to {args.csv_out}")


if __name__ == "__main__":
    main()
    '''
Example command:
python greedyEval.py --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70\
  --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
  --json_dir /home/sprice/MIS/modelAttempt2_5/mis_results_grouped_v3 \
  --csv_out mis_greedy_results.csv
'''
