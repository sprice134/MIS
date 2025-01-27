#!/usr/bin/env python3

import os
import json
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import random_split
from tools2 import MISGraphDataset, GCNForMIS, EarlyStopping, calculate_baseline_mae
from sklearn.metrics import confusion_matrix, f1_score
import argparse
import random
import numpy as np
import networkx as nx
import csv

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def gather_json_and_edgelist_paths(output_dir, node_counts, removal_percents):
    """
    Gathers all JSON result files and their corresponding edgelist directories.

    Args:
        output_dir (str): Directory where MIS JSON files are stored.
        node_counts (list of int): List of node counts to include.
        removal_percents (list of int): List of removal percentages to include.

    Returns:
        tuple: Two lists containing JSON file paths and corresponding edgelist directories.
    """
    json_paths = []
    edgelist_dirs = []

    for n in node_counts:
        for percent in removal_percents:
            json_filename = f"nodes_{n}_removal_{percent}percent.json"
            json_path = os.path.join(output_dir, json_filename)
            if not os.path.exists(json_path):
                print(f"Warning: JSON file '{json_path}' does not exist. Skipping.")
                continue
            json_paths.append(json_path)

            # Assuming edgelist directories follow the pattern generated earlier
            edgelist_dir = os.path.join("generated_graphs", f"nodes_{n}", f"removal_{percent}percent")
            if not os.path.exists(edgelist_dir):
                print(f"Warning: Edgelist directory '{edgelist_dir}' does not exist. Skipping.")
                continue
            edgelist_dirs.append(edgelist_dir)

    return json_paths, edgelist_dirs


def load_graph_from_edgelist(edgelist_path, num_nodes):
    """
    Loads a graph from an edgelist file.

    Args:
        edgelist_path (str): Path to the edgelist file.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        networkx.Graph: The loaded graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    with open(edgelist_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G


def greedy_mis(G):
    """
    Computes a greedy MIS by iteratively selecting the node with the lowest degree.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        set: The computed MIS as a set of node indices.
    """
    MIS = set()
    G_copy = G.copy()
    while G_copy.number_of_nodes() > 0:
        # Select node with the lowest degree
        min_degree_node = min(G_copy.nodes, key=lambda x: G_copy.degree[x])
        MIS.add(min_degree_node)
        # Remove the node and its neighbors
        neighbors = list(G_copy.neighbors(min_degree_node))
        G_copy.remove_node(min_degree_node)
        G_copy.remove_nodes_from(neighbors)
    return MIS


def augmented_greedy_mis(G, probs, threshold):
    """
    Computes an augmented greedy MIS by first selecting nodes with probability >= threshold.

    Args:
        G (networkx.Graph): The input graph.
        probs (list or np.ndarray): List of probabilities for each node.
        threshold (float): Threshold to select nodes.

    Returns:
        set: The augmented MIS as a set of node indices.
    """
    MIS = set()
    G_copy = G.copy()

    # Create a list of nodes with prob >= threshold, sorted by descending probability
    high_prob_nodes = sorted(
        [node for node, prob in enumerate(probs) if prob >= threshold],
        key=lambda x: probs[x],
        reverse=True
    )

    # Select nodes with prob >= threshold ensuring independence
    for node in high_prob_nodes:
        if node not in G_copy:
            continue
        MIS.add(node)
        neighbors = list(G_copy.neighbors(node))
        G_copy.remove_node(node)
        G_copy.remove_nodes_from(neighbors)

    # Perform standard greedy MIS on the remaining graph
    additional_mis = greedy_mis(G_copy)
    MIS.update(additional_mis)

    return MIS


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained GCN model on MIS data with probabilistic labels."
    )
    parser.add_argument(
        "--node_counts",
        type=int,
        nargs='+',
        default=list(range(10, 55, 5)),  # Nodes 10, 15, ..., 50
        help="List of node counts to include, e.g., --node_counts 10 15 20 25 30 35 40 45 50"
    )
    parser.add_argument(
        "--removal_percents",
        type=int,
        nargs='+',
        default=list(range(15, 90, 5)),  # 15%, 20%, ..., 85%
        help="List of edge removal percentages to include, e.g., --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mis_results_grouped",
        help="Directory where MIS JSON result files are stored."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=128,
        help="Number of hidden channels in the GCN."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=7,
        help="Number of GCN layers."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_model_prob.pth",
        help="Path to load the pre-trained model."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold X for augmented greedy MIS."
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="mis_evaluation_results.csv",
        help="Path to save the evaluation CSV."
    )
    args = parser.parse_args()

    node_counts = args.node_counts
    removal_percents = args.removal_percents
    output_dir = args.output_dir
    BATCH_SIZE = args.batch_size
    HIDDEN_CHANNELS = args.hidden_channels
    NUM_LAYERS = args.num_layers
    MODEL_SAVE_PATH = args.model_save_path
    THRESHOLD_X = args.threshold
    CSV_OUTPUT_PATH = args.csv_output

    # Gather JSON paths and corresponding edgelist directories
    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(
        output_dir=output_dir,
        node_counts=node_counts,
        removal_percents=removal_percents
    )

    if not json_paths or not edgelist_dirs:
        print("Error: No valid JSON files or edgelist directories found. Exiting.")
        return

    print(f"Total JSON files found: {len(json_paths)}")
    print(f"Total edgelist directories found: {len(edgelist_dirs)}")

    # Create the dataset with probabilistic labels
    dataset = MISGraphDataset(
        json_paths=json_paths,
        edgelist_dirs=edgelist_dirs,
        label_type='prob'  # Use probabilistic labels
    )

    if len(dataset) == 0:
        print("Error: Dataset is empty after loading. Exiting.")
        return

    print(f"Total graphs in dataset: {len(dataset)}")

    # Split into train (70%), validation (20%), and test (10%) sets
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    # Edge case: ensure test set is at least 1 if possible
    if n_test == 0 and n_val > 0:
        n_val -= 1
        n_test = 1

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Create data loaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = GeometricDataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size 1 for individual graph processing

    print("\nCalculating baseline MAE (predicting 0.5 for every node):")
    baseline_train_mae = calculate_baseline_mae(train_loader, label_type='prob')
    baseline_val_mae = calculate_baseline_mae(val_loader, label_type='prob')
    baseline_test_mae = calculate_baseline_mae(test_loader, label_type='prob')

    print(f"Baseline Training MAE: {baseline_train_mae:.4f}")
    print(f"Baseline Validation MAE: {baseline_val_mae:.4f}")
    print(f"Baseline Test MAE: {baseline_test_mae:.4f}")

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS
    ).to(device)

    # Load the pre-trained model
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded the pre-trained model from '{MODEL_SAVE_PATH}'.")
    else:
        print(f"Error: Pre-trained model '{MODEL_SAVE_PATH}' not found. Exiting.")
        return

    # Ensure model is in evaluation mode
    model.eval()

    # Prepare CSV file
    csv_headers = ['file_path', 'true_MIS_size', 'greedy_MIS_size', 'augmented_greedy_MIS_size']
    csv_rows = []

    # Collect all test predictions and true labels for confusion matrix
    all_preds = []
    all_labels = []

    print("\nEvaluating on the Test Set and Generating CSV...")

    for idx, data in enumerate(test_loader):
        # Assuming 'file_path' is stored in data.file_path
        # Modify based on actual dataset implementation
        file_path = data.file_path[0] if hasattr(data, 'file_path') else "unknown_path"

        num_nodes = data.num_nodes
        edgelist_dir = data.edgelist_dir[0] if hasattr(data, 'edgelist_dir') else None
        edgelist_path = os.path.join(edgelist_dir, f"graph_{idx}.edgelist") if edgelist_dir else None

        if edgelist_path and os.path.exists(edgelist_path):
            G = load_graph_from_edgelist(edgelist_path, num_nodes)
        else:
            print(f"Warning: Edgelist path '{edgelist_path}' does not exist. Skipping graph.")
            continue

        # Compute standard greedy MIS
        standard_mis = greedy_mis(G)
        standard_mis_size = len(standard_mis)

        # Get true MIS size from data
        # Assuming 'MIS_SIZE' is stored in data.mis_size or similar
        # Modify based on actual dataset implementation
        true_mis_size = data.mis_size[0] if hasattr(data, 'mis_size') else "unknown"

        # Get model probabilities
        data = data.to(device)
        with torch.no_grad():
            out = model(data)  # Shape: [num_nodes]
            probs = torch.sigmoid(out).cpu().numpy()  # Apply sigmoid if model outputs logits

        # Compute augmented greedy MIS using threshold X
        augmented_mis = augmented_greedy_mis(G, probs, THRESHOLD_X)
        augmented_mis_size = len(augmented_mis)

        # Append to CSV rows
        csv_rows.append([file_path, true_mis_size, standard_mis_size, augmented_mis_size])

        # Collect predictions and labels for confusion matrix
        # Binarize based on threshold X for augmented MIS
        # Here, for confusion matrix, we need binary classification: 1 if node is in MIS, else 0
        # Assuming data.y contains binary labels or probabilities
        binary_labels = [1 if label >= 0.5 else 0 for label in data.y.cpu().numpy()]
        binary_preds = [1 if prob >= THRESHOLD_X else 0 for prob in probs]

        all_labels.extend(binary_labels)
        all_preds.extend(binary_preds)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_loader):
            print(f"Processed {idx + 1}/{len(test_loader)} graphs.")

    # Write to CSV
    with open(CSV_OUTPUT_PATH, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)

    print(f"\nEvaluation completed. Results saved to '{CSV_OUTPUT_PATH}'.")

    # Define additional thresholds for confusion matrix
    additional_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresholds = [THRESHOLD_X] + additional_thresholds

    print("\nConfusion Matrix and F1 Score for Different Thresholds:")
    for thresh in thresholds:
        # Binarize predictions and labels based on threshold
        binarized_preds = [1 if pred >= thresh else 0 for pred in all_preds]
        binarized_labels = all_labels  # Already binarized based on true MIS membership

        # Compute confusion matrix
        cm = confusion_matrix(binarized_labels, binarized_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases where one class is missing
            tn = cm[0][0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            fp = cm[0][1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
            fn = cm[1][0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
            tp = cm[1][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

        # Compute Precision, Recall, F1 Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\nThreshold: {thresh:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
    '''
    python modelEval_prob.py \
    --node_counts 10 15 20 25 30 35 40 45 50 \
    --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
    --output_dir mis_results_grouped \
    --batch_size 16 \
    --hidden_channels 64 \
    --num_layers 8 \
    --model_save_path best_model_prob.pth \
    --threshold 0.1612 \
    --csv_output evaluation_results.csv

    '''
