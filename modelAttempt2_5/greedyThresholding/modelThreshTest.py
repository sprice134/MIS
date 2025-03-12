#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.utils import to_networkx
import sys
sys.path.append('../')
from tools4 import MISGraphDataset, GCNForMIS  # adjust these imports as needed

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

def min_degree_greedy(G, candidate_nodes):
    """
    Given a NetworkX graph G and a set of candidate nodes,
    compute an independent set via a min-degree greedy algorithm.
    It works on the induced subgraph of candidate_nodes.
    Returns the independent set (as a set of nodes).
    """
    H = G.subgraph(candidate_nodes).copy()
    independent_set = set()
    while H.number_of_nodes() > 0:
        # Select the node with minimum degree.
        v = min(H.nodes(), key=lambda x: H.degree[x])
        independent_set.add(v)
        neighbors = list(H.neighbors(v))
        H.remove_node(v)
        H.remove_nodes_from(neighbors)
    return independent_set

def evaluate_rmse_for_threshold(threshold, dataset, model, device, label_type):
    """
    For a given threshold, run the model on each graph in the dataset,
    threshold the node predictions (keeping only nodes with probability >= threshold),
    compute an independent set using min-degree greedy on the induced subgraph,
    and compare its size to the true MIS size (stored in each graph's attribute).
    Returns the RMSE for this threshold over all graphs.
    """
    errors = []
    # We iterate over each graph in the dataset.
    for data in dataset:
        data = data.to(device)
        model.eval()
        with torch.no_grad():
            out = model(data)
        # In binary mode, apply sigmoid to get probabilities.
        pred = torch.sigmoid(out) if label_type == 'binary' else out
        pred_np = pred.cpu().numpy().flatten()
        # Candidate nodes: indices with predicted probability >= threshold.
        candidate_nodes = set(np.where(pred_np >= threshold)[0])
        # Convert the PyG data to a NetworkX graph.
        G = to_networkx(data, to_undirected=True)
        # It is possible that some nodes are missing (e.g., isolated nodes). Ensure all nodes are included.
        all_nodes = set(range(data.num_nodes))
        if candidate_nodes == set():
            computed_mis_size = 0
        else:
            # Compute independent set using min degree greedy on the induced subgraph.
            mis_set = min_degree_greedy(G, candidate_nodes)
            computed_mis_size = len(mis_set)
        # Get true MIS size from the attribute MIS_SIZE.
        if hasattr(data, 'MIS_SIZE'):
            true_mis = data.MIS_SIZE
        else:
            raise AttributeError("MIS_SIZE attribute not found in the data object.")

        error = (computed_mis_size - true_mis) ** 2
        errors.append(error)
    mse = np.mean(errors)
    rmse = math.sqrt(mse)
    return rmse

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GCN model on each graph to compute RMSE of MIS sizes at various thresholds."
    )
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--node_counts", type=int, nargs="+", required=True,
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", required=True,
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (set to 1 to process one graph at a time).")
    parser.add_argument("--model_save_path", type=str, default="best_model_binary.pth",
                        help="Path to the saved model.")
    parser.add_argument("--label_type", type=str, default="binary", choices=["binary", "prob"],
                        help="Label type: 'binary' or 'prob'")
    parser.add_argument("--csv_out", type=str, default="rmse_results.csv",
                        help="Filename for output CSV with RMSE per threshold.")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Number of hidden channels in the GCN (should match training).")
    parser.add_argument("--num_layers", type=int, default=7,
                        help="Number of layers in the GCN (should match training).")
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
    
    # Create test dataset from the gathered JSON and edgelist directories.
    test_dataset = MISGraphDataset(json_paths=json_paths, edgelist_dirs=edgelist_dirs, label_type=args.label_type)
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty after loading. Exiting.")
        return
    print(f"Total graphs in test dataset: {len(test_dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model.
    if args.label_type == 'binary':
        model = GCNForMIS(hidden_channels=args.hidden_channels, num_layers=args.num_layers, apply_sigmoid=False).to(device)
    else:
        model = GCNForMIS(hidden_channels=args.hidden_channels, num_layers=args.num_layers, apply_sigmoid=True).to(device)
    
    if os.path.exists(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        print(f"Loaded model from '{args.model_save_path}'.")
    else:
        print(f"Error: Model file '{args.model_save_path}' not found. Exiting.")
        return
    
    # Define thresholds: 0, 0.05, 0.10, ... 0.95, 1.0.
    thresholds = [0.0] + [round(x * 0.05, 2) for x in range(1, 20)] + [1.0]
    
    # Evaluate RMSE for each threshold.
    rmse_results = []
    for thr in thresholds:
        rmse = evaluate_rmse_for_threshold(thr, test_dataset, model, device, args.label_type)
        rmse_results.append({"threshold": thr, "rmse": rmse})
        print(f"Threshold {thr:.2f}: RMSE = {rmse:.4f}")
    
    # Save results to CSV.
    df = pd.DataFrame(rmse_results)
    df.to_csv(args.csv_out, index=False)
    print(f"RMSE results saved to {args.csv_out}")

if __name__ == "__main__":
    main()

    '''
    Example command:
    python modelThreshTest.py \
       --json_dir /home/sprice/MIS/modelAttempt2_5/test_mis_results_grouped_v3 \
       --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 \
       --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
       --batch_size 1 \
       --model_save_path ../best_model_binary_32_176_28_0.001_v1.pth \
       --label_type binary \
       --csv_out rmse_results.csv \
       --hidden_channels 176 \
       --num_layers 28
    '''
