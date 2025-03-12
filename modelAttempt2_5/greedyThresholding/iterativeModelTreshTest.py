#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import argparse
import math
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.utils import to_networkx, from_networkx
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def min_degree_greedy(subgraph, candidate_nodes):
    """
    Given a NetworkX subgraph and a set of candidate nodes,
    select the candidate node with the minimum degree in the subgraph.
    """
    chosen = min(candidate_nodes, key=lambda n: subgraph.degree[n])
    return chosen

def iterative_mis_selection(data, model, device, threshold, label_type):
    """
    Given a PyG data object, repeatedly run the model on the induced subgraph of remaining nodes.
    At each iteration:
      - Run model inference on the current subgraph.
      - Threshold predictions: keep nodes with probability >= threshold.
      - If candidates exist, select one using the min-degree rule.
      - Remove that node and its neighbors from the current set.
    Returns the computed MIS (a set of node indices referring to the original graph).
    """
    remaining_nodes = set(range(data.num_nodes))
    mis_set = set()
    
    while remaining_nodes:
        # Create an induced subgraph on the remaining nodes.
        full_G = to_networkx(data, to_undirected=True)
        subG = full_G.subgraph(remaining_nodes).copy()
        
        # Create a PyG Data object from the subgraph.
        sub_data = from_networkx(subG)
        # Ensure node features exist.
        if (not hasattr(sub_data, 'x')) or (sub_data.x is None):
            # Use the number of nodes from the subgraph (should equal len(subG.nodes()))
            num_sub_nodes = subG.number_of_nodes()
            sub_data.x = torch.ones((num_sub_nodes, 1), dtype=torch.float)
        sub_data = sub_data.to(device)
        
        model.eval()
        with torch.no_grad():
            out = model(sub_data)
        # Get predictions.
        pred = torch.sigmoid(out) if label_type == 'binary' else out
        pred_np = pred.cpu().numpy().flatten()
        
        # from_networkx returns nodes in sorted order.
        sub_nodes = sorted(list(remaining_nodes))
        # Determine candidate nodes (those with prediction >= threshold).
        candidate_indices = [i for i, p in enumerate(pred_np) if p >= threshold]
        candidate_nodes = {sub_nodes[i] for i in candidate_indices}
        
        # If no candidate nodes remain, break.
        if not candidate_nodes:
            break
        
        # Select one candidate using min-degree rule.
        chosen = min_degree_greedy(subG, candidate_nodes)
        mis_set.add(chosen)
        # Remove chosen node and its neighbors from remaining nodes.
        neighbors = set(subG.neighbors(chosen))
        remaining_nodes -= (neighbors | {chosen})
    
    return mis_set

def evaluate_rmse_iterative(threshold, dataset, model, device, label_type):
    """
    For a given threshold, run the iterative MIS selection on each graph in the dataset,
    then compare the computed MIS size to the true MIS size stored in each graph.
    Returns the RMSE over the dataset.
    """
    errors = []
    for data in dataset:
        data = data.to(device)
        computed_mis = iterative_mis_selection(data, model, device, threshold, label_type)
        computed_mis_size = len(computed_mis)
        
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
        description="Iteratively select MIS nodes using model inference and min-degree greedy selection."
    )
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--node_counts", type=int, nargs="+", required=True,
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", required=True,
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--model_save_path", type=str, default="best_model_binary.pth",
                        help="Path to the saved model.")
    parser.add_argument("--label_type", type=str, default="binary", choices=["binary", "prob"],
                        help="Label type: 'binary' or 'prob'")
    parser.add_argument("--csv_out", type=str, default="iterative_rmse_results.csv",
                        help="Filename for output CSV with RMSE per threshold.")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Number of hidden channels in the GCN (should match training).")
    parser.add_argument("--num_layers", type=int, default=7,
                        help="Number of layers in the GCN (should match training).")
    # Default thresholds from 0.0 to 1.0 in steps of 0.05.
    default_thresholds = [round(x * 0.05, 2) for x in range(0, 21)]
    parser.add_argument("--thresholds", type=float, nargs="+", default=default_thresholds,
                        help="List of threshold values to test, e.g. 0.0 0.05 ... 1.0")
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

    # Evaluate RMSE for each threshold in parallel.
    results = []
    with ThreadPoolExecutor(max_workers=len(args.thresholds)) as executor:
        future_to_thr = {
            executor.submit(evaluate_rmse_iterative, thr, test_dataset, model, device, args.label_type): thr 
            for thr in args.thresholds
        }
        for future in as_completed(future_to_thr):
            thr = future_to_thr[future]
            try:
                rmse = future.result()
                results.append({"threshold": thr, "rmse": rmse})
                print(f"Threshold {thr:.2f}: Iterative RMSE = {rmse:.4f}")
            except Exception as exc:
                print(f"Threshold {thr:.2f} generated an exception: {exc}")
    
    # Save results to CSV.
    df = pd.DataFrame(results)
    df.to_csv(args.csv_out, index=False)
    print(f"Iterative RMSE results saved to {args.csv_out}")

if __name__ == "__main__":
    main()


    '''
    Example command:
    python iterativeModelTreshTest.py \
       --json_dir /home/sprice/MIS/modelAttempt2_5/test_mis_results_grouped_v3 \
       --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 \
       --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
       --model_save_path ../best_model_binary_32_176_28_0.001_v1.pth \
       --label_type binary \
       --csv_out rmse_results_iterative.csv \
       --hidden_channels 176 \
       --num_layers 28
    '''