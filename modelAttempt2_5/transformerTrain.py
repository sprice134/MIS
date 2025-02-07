#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import argparse
import random
import numpy as np
import optuna

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import add_self_loops

from sklearn.metrics import confusion_matrix

# ------------------------------
# MISGraphDataset (as before)
# ------------------------------
class MISGraphDataset(Dataset):
    """
    A dataset that:
    - Reads multiple JSON files containing, for each graph, its "file_path" and either "MIS_CELLS" or "MIS_CELLS_PROB".
    - Loads the corresponding .edgelist files from multiple directories.
    - Produces a PyTorch Geometric `Data` object with:
        - x: node features (all ones) [num_nodes, 1]
        - edge_index: shape [2, num_edges], undirected, with self-loops added.
        - y: labels [num_nodes] (0/1 or float in [0,1])
    """
    def __init__(self, json_paths, edgelist_dirs, label_type='binary'):
        super().__init__()
        self.inference_mode = not bool(json_paths)
        if self.inference_mode:
            self.file_paths = edgelist_dirs  # directly .edgelist paths
            self.labels_dict = {path: None for path in self.file_paths}
            self.label_type = None
        else:
            if len(json_paths) != len(edgelist_dirs):
                raise AssertionError("JSON_PATHS and EDGELIST_DIRS must have the same length.")
            self.file_paths = []
            self.labels_dict = {}
            self.label_type = label_type.lower()
            if self.label_type not in ['binary', 'prob']:
                raise ValueError("label_type must be 'binary' or 'prob'.")
            for json_path, edgelist_dir in zip(json_paths, edgelist_dirs):
                if not os.path.exists(json_path):
                    print(f"Warning: JSON file '{json_path}' does not exist. Skipping.")
                    continue
                with open(json_path, 'r') as f:
                    try:
                        mis_data = json.load(f)  # a list of dicts
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON file '{json_path}': {e}")
                        continue
                for entry in mis_data:
                    base_name = os.path.basename(entry.get("file_path", ""))
                    if not base_name:
                        print(f"Warning: Missing 'file_path' in entry {entry}. Skipping.")
                        continue
                    full_edgelist_path = os.path.join(edgelist_dir, base_name)
                    label_key = "MIS_CELLS" if self.label_type == 'binary' else "MIS_CELLS_PROB"
                    labels = entry.get(label_key, [])
                    if not os.path.exists(full_edgelist_path):
                        print(f"Warning: Edgelist file '{full_edgelist_path}' does not exist. Skipping.")
                        continue
                    if not isinstance(labels, list):
                        print(f"Warning: '{label_key}' for '{full_edgelist_path}' is not a list. Skipping.")
                        continue
                    self.file_paths.append(full_edgelist_path)
                    self.labels_dict[full_edgelist_path] = labels
            self.file_paths.sort()
            if not self.file_paths:
                print("Warning: No valid .edgelist files found across all JSONs/directories!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph_path = self.file_paths[idx]
        edges = []
        max_node_id = 0
        with open(graph_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) == 2:
                    try:
                        u, v = int(tokens[0]), int(tokens[1])
                        edges.append((u, v))
                        edges.append((v, u))  # undirected edge
                        max_node_id = max(max_node_id, u, v)
                    except ValueError:
                        print(f"Warning: Invalid edge line '{line}' (line {line_num}) in '{graph_path}'. Skipping.")
                elif len(tokens) == 1:
                    try:
                        u = int(tokens[0])
                        max_node_id = max(max_node_id, u)
                    except ValueError:
                        print(f"Warning: Invalid node line '{line}' (line {line_num}) in '{graph_path}'. Skipping.")
                else:
                    print(f"Warning: Unexpected format in line '{line}' (line {line_num}) in '{graph_path}'. Skipping.")
        num_nodes = max_node_id + 1
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        x = torch.ones((num_nodes, 1), dtype=torch.float)
        if self.inference_mode:
            y = torch.zeros(num_nodes, dtype=torch.float)
        else:
            full_label_array = self.labels_dict.get(graph_path, [])
            if len(full_label_array) < num_nodes:
                label_name = "MIS_CELLS" if self.label_type == 'binary' else "MIS_CELLS_PROB"
                padding_needed = num_nodes - len(full_label_array)
                print(f"Warning: '{label_name}' length ({len(full_label_array)}) < expected ({num_nodes}) for file '{graph_path}'. Padding with zeros.")
                if self.label_type == 'binary':
                    full_label_array += [0] * padding_needed
                else:
                    full_label_array += [0.0] * padding_needed
            full_label_array = full_label_array[:num_nodes]
            y = torch.tensor(full_label_array, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

# ------------------------------
# Graph Transformer Model with LayerNorm
# ------------------------------
class GraphTransformer(nn.Module):
    """
    A graph transformer model using TransformerConv layers.
    It embeds node features, applies several TransformerConv layers with dropout and layer normalization,
    and outputs a per-node probability.
    """
    def __init__(self, in_channels=1, hidden_channels=128, num_layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(TransformerConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              heads=heads,
                                              concat=False,
                                              dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.out_lin = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_lin(x)
        return torch.sigmoid(x).squeeze(-1)

# ------------------------------
# Training / Evaluation Functions
# ------------------------------
def train(model, loader, optimizer, device, label_type='binary'):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, label_type='binary'):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        total_loss += loss.item()
    return total_loss / len(loader)

def gather_json_and_edgelist_paths(output_dir, node_counts, removal_percents):
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
            edgelist_dir = os.path.join("generated_graphs", f"nodes_{n}", f"removal_{percent}percent")
            if not os.path.exists(edgelist_dir):
                print(f"Warning: Edgelist directory '{edgelist_dir}' does not exist. Skipping.")
                continue
            edgelist_dirs.append(edgelist_dir)
    return json_paths, edgelist_dirs

# ------------------------------
# Objective Function for Hyperparameter Optimization
# ------------------------------
def objective(trial):
    # Hyperparameters to tune:
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    heads = trial.suggest_int("heads", 1, 8)

    # For a faster tuning run, you may use a subset of your dataset:
    output_dir = args.output_dir  # use the argument passed to the script
    node_counts = args.node_counts
    removal_percents = args.removal_percents
    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(output_dir, node_counts, removal_percents)
    dataset = MISGraphDataset(json_paths=json_paths, edgelist_dirs=edgelist_dirs, label_type=args.label_type)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after loading.")
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    if n_test == 0 and n_val > 0:
        n_val -= 1
        n_test = 1
    train_dataset, val_dataset, _ = random_split(dataset, [n_train, n_val, n_test],
                                                   generator=torch.Generator().manual_seed(42))
    train_loader = GeometricDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphTransformer(in_channels=1,
                             hidden_channels=hidden_channels,
                             num_layers=num_layers,
                             heads=heads,
                             dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=False)

    best_val_loss = float('inf')
    epochs = args.epochs // 10  # For tuning, you may reduce the number of epochs
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, label_type=args.label_type)
        val_loss = evaluate(model, val_loader, device, label_type=args.label_type)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience // 2:
            break
    return best_val_loss

# ------------------------------
# Main Script: Parse Args and Run Hyperparameter Tuning
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Graph Transformer on MIS graphs using Optuna."
    )
    parser.add_argument("--node_counts", type=int, nargs='+', default=list(range(10, 55, 5)),
                        help="List of node counts to include.")
    parser.add_argument("--removal_percents", type=int, nargs='+', default=list(range(15, 90, 5)),
                        help="List of removal percentages to include.")
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped",
                        help="Directory where MIS JSON result files are stored.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for full training.")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping during tuning.")
    parser.add_argument("--label_type", type=str, default="prob",
                        help="Label type: 'binary' or 'prob'.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run.")
    args = parser.parse_args()

    # Run Optuna study:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("  Best Validation Loss: {:.4f}".format(trial.value))
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Once the best hyperparameters are found, you can re-train the model on the full training set
    # (or even on train+validation) and then test on the held-out test set.
    # Here is an example of retraining:

    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(args.output_dir, args.node_counts, args.removal_percents)
    dataset = MISGraphDataset(json_paths=json_paths, edgelist_dirs=edgelist_dirs, label_type=args.label_type)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)  # train on 80% of data (train + val)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test],
                                               generator=torch.Generator().manual_seed(42))
    train_loader = GeometricDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = GeometricDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_params = trial.params

    best_model = GraphTransformer(
        in_channels=1,
        hidden_channels=best_params["hidden_channels"],
        num_layers=best_params["num_layers"],
        heads=best_params["heads"],
        dropout=best_params["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(best_model, train_loader, optimizer, device, label_type=args.label_type)
        val_loss = evaluate(best_model, test_loader, device, label_type=args.label_type)
        scheduler.step(val_loss)
        print(f"Retrain Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(best_model.state_dict(), "best_model_overall.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping triggered during retraining.")
            break

    print("Best overall model saved as 'best_model_overall.pth'.")
