#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader as GeometricDataLoader

class MISGraphDataset(Dataset):
    """
    A dataset that:
    - Reads multiple JSON files containing, for each graph, its "file_path" and either "MIS_CELLS" or "MIS_CELLS_PROB".
    - Loads the corresponding .edgelist files from multiple directories.
    - Produces a PyTorch Geometric `Data` object with:
        - x: node features (all ones) [num_nodes, 1]
        - edge_index: shape [2, num_edges], undirected
        - y: labels [num_nodes] (0/1 for binary, or floats in [0,1] for probability labels)
    - Skips reindexing: we assume node IDs are valid in [0..max_node].
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
            # Added dictionary to store MIS_SIZE values.
            self.mis_size_dict = {}
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
                    # Save MIS_SIZE from JSON.
                    self.mis_size_dict[full_edgelist_path] = entry.get("MIS_SIZE")
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
                        edges.append((v, u))  # undirected
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
        data = Data(x=x, edge_index=edge_index, y=y)
        # Attach the MIS_SIZE value from JSON.
        data.MIS_SIZE = self.mis_size_dict.get(graph_path)
        return data


class GCNForMIS(nn.Module):
    """
    A simple GCN for node-level classification.
    The behavior is configurable via apply_sigmoid:
    - If apply_sigmoid is False (binary mode), the forward returns raw logits.
    - If apply_sigmoid is True (prob mode), the forward returns probabilities.
    """
    def __init__(self, hidden_channels=16, num_layers=2, apply_sigmoid=False):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels=1, out_channels=hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.out_conv = GCNConv(in_channels=hidden_channels, out_channels=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.out_conv(x, edge_index).squeeze(-1)
        if self.apply_sigmoid:
            return torch.sigmoid(x)
        else:
            return x  # Raw logits


class EarlyStopping:
    """
    Early stops training if validation loss doesn't improve after a given patience.
    Tracks the best validation MAE.
    """
    def __init__(self, patience=20, verbose=False, delta=0.0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.best_mae = None
        self.early_stop = False

    def __call__(self, val_loss, val_mae, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_mae = val_mae
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_mae = val_mae
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if self.verbose:
                print("Validation loss decreased. Resetting patience.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased to {val_loss:.4f}. Saving model to {self.path}.")


def calculate_baseline_mae(loader, label_type='prob'):
    """
    Calculates the MAE for a baseline that predicts 0.5 for every node.
    """
    total_mae = 0
    total_nodes = 0
    for data in loader:
        baseline_preds = torch.full_like(data.y, 0.5)
        mae = torch.abs(baseline_preds - data.y).sum().item()
        total_mae += mae
        total_nodes += data.y.numel()
    return total_mae / total_nodes if total_nodes > 0 else 0.0
