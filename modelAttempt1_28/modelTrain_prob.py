#!/usr/bin/env python3

import os
import json
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tools3 import MISGraphDataset, GCNForMIS, EarlyStopping, train, evaluate, calculate_baseline_mae
from sklearn.metrics import confusion_matrix
import argparse
import random
import numpy as np

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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a GCN model on Maximum Independent Set (MIS) data with probabilistic labels."
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
        help="Batch size for training."
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
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopping."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_model_prob.pth",
        help="Path to save the best model."
    )
    args = parser.parse_args()

    node_counts = args.node_counts
    removal_percents = args.removal_percents
    output_dir = args.output_dir
    BATCH_SIZE = args.batch_size
    HIDDEN_CHANNELS = args.hidden_channels
    NUM_LAYERS = args.num_layers
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    PATIENCE = args.patience
    MODEL_SAVE_PATH = args.model_save_path

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
    test_loader = GeometricDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=MODEL_SAVE_PATH)

    # Training loop with Early Stopping
    best_validation_mae = None  # Initialize variable to store best validation MAE
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, device, label_type='prob')
        val_loss, val_mae = evaluate(model, val_loader, device, label_type='prob')

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | "
              f"Validation MAE: {val_mae:.4f}")

        # Early Stopping check
        early_stopping(val_loss, val_mae, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            best_validation_mae = early_stopping.best_mae  # Retrieve best validation MAE
            break

    # Load the best model
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded the best model from '{MODEL_SAVE_PATH}'.")
    else:
        print(f"Warning: '{MODEL_SAVE_PATH}' not found. Using the current model.")

    # Evaluate on Test Set
    test_loss, test_mae = evaluate(model, test_loader, device, label_type='prob')
    print(f"\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    # Collect all test predictions and true labels
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)  # Shape: [num_nodes]
            all_preds.extend(out.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # Define thresholds: best_validation_mae and [0.1, 0.2, 0.3, 0.4, 0.5]
    thresholds = [best_validation_mae if best_validation_mae is not None else 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]

    print("\nConfusion Matrix and F1 Score for Different Thresholds:")
    for thresh in thresholds:
        # Binarize predictions and labels based on threshold
        binarized_preds = [1 if pred >= thresh else 0 for pred in all_preds]
        binarized_labels = [1 if label >= 0.5 else 0 for label in all_labels]  # Assuming labels are 0 or fractions

        # Compute confusion matrix
        cm = confusion_matrix(binarized_labels, binarized_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Compute F1 Score
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            precision = recall = f1 = 0.0

        print(f"\nThreshold: {thresh:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")



if __name__ == "__main__":
    main()
    '''
    Example command to run the script:
    python modelTrain_prob.py \
        --node_counts 10 15 20 25 30 35 40 45 50\
        --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
        --output_dir mis_results_grouped \
        --batch_size 16\
        --hidden_channels 64 \
        --num_layers 8 \
        --learning_rate 0.005 \
        --epochs 1000 \
        --patience 20 \
        --model_save_path best_model_prob.pth

    python modelTrain_prob.py \
        --node_counts 10 15 20 25 30 35 40 \
        --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
        --output_dir mis_results_grouped_v2 \
        --batch_size 16\
        --hidden_channels 64 \
        --num_layers 8 \
        --learning_rate 0.005 \
        --epochs 1000 \
        --patience 20 \
        --model_save_path best_model_prob.pth
    '''
