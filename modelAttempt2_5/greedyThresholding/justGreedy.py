#!/usr/bin/env python3
import os
import math
import json
import random
import re
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx

# Add the parent directory to import MISGraphDataset from tools4.
sys.path.append('../')
from tools4 import MISGraphDataset  # adjust these imports as needed

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

def greedy_MIS(G_sub):
    """
    Compute an independent set using a greedy algorithm on graph G_sub.
    This algorithm repeatedly selects a vertex (here, the one with minimum degree),
    adds it to the independent set, and removes it along with its neighbors.
    Returns the independent set (as a set of nodes).
    """
    I = set()
    remaining = set(G_sub.nodes())
    while remaining:
        # Select the vertex with minimum degree in the remaining subgraph.
        v = min(remaining, key=lambda x: G_sub.degree(x))
        I.add(v)
        # Remove v and its neighbors.
        neighbors = set(G_sub.neighbors(v))
        remaining.remove(v)
        remaining -= neighbors
    return I

def evaluate_rmse_greedy(dataset):
    """
    For each graph in the dataset, compute the greedy MIS (on the full graph) and compare
    its size to the true MIS size stored in the graph's attribute MIS_SIZE.
    Returns the RMSE over all graphs.
    """
    errors = []
    # Iterate over each graph in the dataset.
    for data in dataset:
        # Convert the PyG data to a NetworkX graph.
        G = nx.convert_matrix.from_scipy_sparse_matrix(data.edge_index) if hasattr(data, 'edge_index') else nx.Graph()
        # Alternatively, if a conversion utility is provided, use that.
        # For simplicity, we assume the dataset graphs can be converted via:
        G = nx.from_scipy_sparse_matrix(data.edge_index) if hasattr(data, 'edge_index') else nx.Graph()
        # Ensure all nodes are included.
        all_nodes = set(range(data.num_nodes))
        if G.number_of_nodes() < data.num_nodes:
            G.add_nodes_from(all_nodes)
        # Compute the greedy MIS on the full graph.
        mis_set = greedy_MIS(G)
        computed_mis_size = len(mis_set)
        # Get true MIS size from the attribute MIS_SIZE.
        if hasattr(data, 'MIS_SIZE'):
            true_mis = data.MIS_SIZE
        else:
            raise AttributeError("MIS_SIZE attribute not found in the data object.")
        error = (computed_mis_size - true_mis) ** 2
        errors.append(error)
        print(f"Graph: computed MIS size = {computed_mis_size}, true MIS size = {true_mis}")
    mse = np.mean(errors)
    rmse = math.sqrt(mse)
    return rmse

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RMSE of Greedy MIS (ignoring model predictions) on a set of graphs."
    )
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--node_counts", type=int, nargs="+", required=True,
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", required=True,
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--csv_out", type=str, default="rmse_results_greedy.csv",
                        help="Filename for output CSV with RMSE value.")
    args = parser.parse_args()
    
    # Gather JSON paths and corresponding edgelist directories.
    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(
        json_dir=args.json_dir,
        node_counts=args.node_counts,
        removal_percents=args.removal_percents
    )
    
    if not json_paths or not edgelist_dirs:
        print("Error: No valid JSON files or edgelist directories found. Exiting.")
        return
    
    print(f"Total JSON files found: {len(json_paths)}")
    print(f"Total edgelist directories found: {len(edgelist_dirs)}")
    
    # Create the test dataset.
    test_dataset = MISGraphDataset(json_paths=json_paths, edgelist_dirs=edgelist_dirs)
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty after loading. Exiting.")
        return
    print(f"Total graphs in test dataset: {len(test_dataset)}")
    
    # Evaluate RMSE using greedy MIS on the full graphs.
    rmse = evaluate_rmse_greedy(test_dataset)
    print(f"\nRMSE of greedy MIS over the dataset: {rmse:.4f}")
    
    # Save the RMSE result to a CSV file.
    df = pd.DataFrame([{"rmse": rmse}])
    df.to_csv(args.csv_out, index=False)
    print(f"RMSE results saved to {args.csv_out}")

if __name__ == "__main__":
    main()


'''
python justGreedy.py \
       --json_dir /home/sprice/MIS/modelAttempt2_5/test_mis_results_grouped_v3 \
       --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 \
       --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
       --csv_out rmse_greedy_results.csv 

'''