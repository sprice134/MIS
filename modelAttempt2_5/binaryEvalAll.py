#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import argparse
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from concurrent.futures import ProcessPoolExecutor
from tools4 import MISGraphDataset, GCNForMIS  # adjust these imports as needed

# --- Updated JSON/Edgelist Access Functions ---
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

# --- Prediction and Evaluation Functions ---
def get_predictions(model, loader, device, label_type):
    """
    Run the model over the test set once to get predictions and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            # In binary mode, apply sigmoid to get probabilities.
            preds = torch.sigmoid(out) if label_type == 'binary' else out
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(data.y.cpu().numpy().flatten())
    return np.array(all_preds), np.array(all_labels)

def compute_metrics(threshold, preds, labels):
    """
    Binarize predictions using the given threshold (for preds) and a fixed threshold of 0.5 for labels.
    Compute confusion matrix, precision, recall, F1 and accuracy.
    """
    binary_preds = (preds >= threshold).astype(int)
    binary_labels = (labels >= 0.5).astype(int)
    
    cm = confusion_matrix(binary_labels, binary_preds)
    tn = fp = fn = tp = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(binary_labels, binary_preds, zero_division=0)
    recall = recall_score(binary_labels, binary_preds, zero_division=0)
    f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    accuracy = accuracy_score(binary_labels, binary_preds)
    
    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GCN model on a test set at multiple thresholds."
    )
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--node_counts", type=int, nargs="+", required=True,
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", required=True,
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the test set.")
    parser.add_argument("--model_save_path", type=str, default="best_model_binary.pth",
                        help="Path to the saved model.")
    parser.add_argument("--label_type", type=str, default="binary", choices=["binary", "prob"],
                        help="Label type: 'binary' or 'prob'")
    parser.add_argument("--csv_out", type=str, default="evaluation_metrics.csv",
                        help="Filename for output CSV with evaluation metrics.")
    # Optionally include model architecture parameters (if needed to instantiate the model)
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Number of hidden channels in the GCN (should match training).")
    parser.add_argument("--num_layers", type=int, default=7,
                        help="Number of layers in the GCN (should match training).")
    args = parser.parse_args()
    
    # Gather JSON paths and corresponding edgelist directories using the updated function.
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
    
    test_loader = GeometricDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    
    # Get predictions once over the entire test set.
    preds, labels = get_predictions(model, test_loader, device, args.label_type)
    print("Completed predictions on test set.")
    
    # Define thresholds: 0, 0.05, 0.10, ... 0.95, 1.0.
    thresholds = [0.0] + [round(x * 0.05, 2) for x in range(1, 20)] + [1.0]
    
    # Parallel evaluation of metrics.
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_metrics, thr, preds, labels) for thr in thresholds]
        for future in futures:
            results.append(future.result())
    
    # Save the results to CSV.
    df = pd.DataFrame(results)
    df.to_csv(args.csv_out, index=False)
    print(f"Evaluation metrics saved to {args.csv_out}")

if __name__ == "__main__":
    main()
    '''
    python binaryEvalAll.py \
    --json_dir /home/sprice/MIS/modelAttempt2_5/test_mis_results_grouped_v3 \
    --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 \
    --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
    --batch_size 32 \
    --model_save_path best_model_binary_32_176_28_0.001_v1.pth \
    --label_type binary \
    --csv_out binary_evaluation_metrics.csv \
    --hidden_channels 176 \
    --num_layers 28
    '''