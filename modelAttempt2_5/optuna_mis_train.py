#!/usr/bin/env python3
"""
This script uses Optuna to optimize the hyperparameters for training the
GCN model on MIS data with probabilistic labels. The hyperparameters being
tuned are:
  - batch_size
  - hidden_channels
  - num_layers
  - learning_rate

The training data is loaded using MISGraphDataset from the provided JSON and
edgelist directories. The objective function trains the model with early
stopping and returns the validation MAE.

Usage:
    python optuna_mis_train.py --output_dir mis_results_grouped ...
"""

import os
import json
import torch
import random
import numpy as np
import argparse
import optuna

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tools3 import (
    MISGraphDataset,
    GCNForMIS,
    EarlyStopping,
    train,
    evaluate,
    calculate_baseline_mae
)

# Set seeds for reproducibility
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


def objective(trial):
    # Define hyperparameters to optimize.
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_channels = trial.suggest_int("hidden_channels", 64, 256, step=4)
    num_layers = trial.suggest_int("num_layers", 4, 64, step=8)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # Other training settings (you can adjust these as needed)
    epochs = 1000
    patience = 25

    # Hard-code or parse additional arguments for dataset location
    node_counts = list(range(5, 80, 5))  # e.g., 10, 15, ... 50
    removal_percents = list(range(15, 90, 5))  # e.g., 15, 20, ... 85

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped_v3",
                        help="Directory where MIS JSON result files are stored.")
    args, _ = parser.parse_known_args()  # Use default or pass arguments

    output_dir = args.output_dir

    # Gather JSON and edgelist paths.
    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(
        output_dir=output_dir,
        node_counts=node_counts,
        removal_percents=removal_percents
    )

    if not json_paths or not edgelist_dirs:
        raise RuntimeError("No valid JSON files or edgelist directories found.")

    # Create the dataset (using probabilistic labels)
    dataset = MISGraphDataset(
        json_paths=json_paths,
        edgelist_dirs=edgelist_dirs,
        label_type='prob'
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after loading.")

    # Split dataset into train (70%), validation (20%), and test (10%).
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    if n_test == 0 and n_val > 0:
        n_val -= 1
        n_test = 1

    train_dataset, val_dataset, _ = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders for train and validation sets.
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with current hyperparameters.
    model = GCNForMIS(
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Use a temporary model save path (overwritten each trial)
    model_save_path = "temp_model_trial.pth"
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=model_save_path)

    # Training loop with early stopping.
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, label_type='prob')
        val_loss, val_mae = evaluate(model, val_loader, device, label_type='prob')
        # Report intermediate value to Optuna so that pruning can be applied.
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        early_stopping(val_loss, val_mae, model)
        if early_stopping.early_stop:
            break

    # Load the best model from early stopping.
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("Warning: No model file found from early stopping.")

    # Evaluate on the validation set.
    _, val_mae = evaluate(model, val_loader, device, label_type='prob')
    return val_mae


def main():
    parser = argparse.ArgumentParser(
        description="Optimize GCN hyperparameters for MIS training with Optuna."
    )
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials.")
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped",
                        help="Directory where MIS JSON result files are stored.")
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value (Validation MAE): ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
    '''
    python optuna_mis_train.py --n_trials 25 --output_dir mis_results_grouped_v3
'''