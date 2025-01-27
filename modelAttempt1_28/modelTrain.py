#!/usr/bin/env python3

import os
import json
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tools import MISGraphDataset, GCNForMIS, EarlyStopping, train, evaluate
from sklearn.metrics import confusion_matrix
import random
import numpy as np

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
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a GCN model on Maximum Independent Set (MIS) data."
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
        default="best_model.pth",
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

    # Create the dataset
    dataset = MISGraphDataset(
        json_paths=json_paths,
        edgelist_dirs=edgelist_dirs
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
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | "
              f"Validation Acc: {val_acc:.4f}")

        # Early Stopping check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Load the best model
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded the best model from '{MODEL_SAVE_PATH}'.")
    else:
        print(f"Warning: '{MODEL_SAVE_PATH}' not found. Using the current model.")

    # Evaluate on Test Set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Example inference on test set with Confusion Matrix
    # print("\nInference on the test set and Confusion Matrix:")
    model.eval()
    all_preds = []
    all_labels = []
    for i, data in enumerate(test_loader):
        data = data.to(device)
        out = model(data)  # Shape: [num_nodes]
        preds = (out >= 0.5).float().cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        # print(f"Graph batch {i}: predicted (sigmoid) = {preds}, labels = {labels}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    if cm.shape == (2, 2):  # Ensure both classes are present
        print("\nConfusion Matrix Breakdown:")
        print(f"True Negatives: {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives: {cm[1][1]}")
    else:
        print("Warning: Confusion matrix is not 2x2 (some class missing in test set).")

if __name__ == "__main__":
    main()
    #python3 train_model.py \
        # --node_counts 10 15 20 25 30 35 40 45 50 \
        # --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
        # --output_dir mis_results_grouped \
        # --batch_size 16 \
        # --hidden_channels 128 \
        # --num_layers 7 \
        # --learning_rate 0.001 \
        # --epochs 1000 \
        # --patience 20 \
        # --model_save_path best_model.pth
    '''
    python3 modelTrain.py \
        --node_counts 10 15 20 25 30 35 40 45 50\
        --output_dir mis_results_grouped \
        --batch_size 8 \
        --hidden_channels 64 \
        --num_layers 5 \
        --learning_rate 0.005 \
        --epochs 1000 \
        --patience 20 \
        --model_save_path best_model.pth
    '''

