import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import confusion_matrix


class MISGraphDataset(Dataset):
    """
    A dataset that:
    - Reads multiple JSON files containing, for each graph, its "file_path" and either "MIS_CELLS" or "MIS_CELLS_PROB".
    - Loads the corresponding .edgelist files from multiple directories.
    - Produces a PyTorch Geometric `Data` object with:
        - x: node features (all ones) [num_nodes, 1]
        - edge_index: shape [2, num_edges], undirected
        - y: labels [num_nodes] (0/1 or float in [0,1])
    - Skips reindexing: we assume node IDs are valid in [0..max_node].
    """

    def __init__(self, json_paths, edgelist_dirs, label_type='binary'):
        """
        Args:
            json_paths (list of str): List of JSON file paths containing MIS results.
            edgelist_dirs (list of str): List of directories or file paths containing .edgelist files.
            label_type (str): 'binary' or 'prob', or None for inference-only mode.
        """
        super().__init__()

        # Determine if the dataset is in inference mode (no labels)
        self.inference_mode = not bool(json_paths)

        if self.inference_mode:
            # Inference mode: No labels provided
            self.file_paths = edgelist_dirs  # directly .edgelist paths
            self.labels_dict = {path: None for path in self.file_paths}
            self.label_type = None
        else:
            # Training/Validation/Test mode
            if len(json_paths) != len(edgelist_dirs):
                raise AssertionError(
                    "JSON_PATHS and EDGELIST_DIRS must have the same length."
                )
            self.file_paths = []
            self.labels_dict = {}
            self.label_type = label_type.lower()
            if self.label_type not in ['binary', 'prob']:
                raise ValueError("label_type must be 'binary' or 'prob'.")

            # Build file_paths & labels_dict
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

                # For each entry in mis_data
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

            # Sort for reproducibility
            self.file_paths.sort()
            if not self.file_paths:
                print("Warning: No valid .edgelist files found across all JSONs/directories!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph_path = self.file_paths[idx]

        edges = []
        node_ids = set()
        max_node_id = 0

        # 1) Read the .edgelist file (skip reindexing)
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
                        node_ids.update([u, v])
                        max_node_id = max(max_node_id, u, v)
                    except ValueError:
                        print(f"Warning: Invalid edge line '{line}' (line {line_num}) in '{graph_path}'. Skipping.")
                elif len(tokens) == 1:
                    # Single-token => isolated node
                    try:
                        u = int(tokens[0])
                        node_ids.add(u)
                        max_node_id = max(max_node_id, u)
                    except ValueError:
                        print(f"Warning: Invalid node line '{line}' (line {line_num}) in '{graph_path}'. Skipping.")
                else:
                    print(f"Warning: Unexpected format in line '{line}' (line {line_num}) in '{graph_path}'. Skipping.")

        # 2) We skip any reindexing. We assume node IDs are 0..max_node_id
        num_nodes = max_node_id + 1

        # 3) Build edge_index
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 4) Node features
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        # 5) Build y
        if self.inference_mode:
            y = torch.zeros(num_nodes, dtype=torch.float)
        else:
            full_label_array = self.labels_dict.get(graph_path, [])
            # Ensure we have enough labels for [0..max_node_id]
            if len(full_label_array) < num_nodes:
                label_name = "MIS_CELLS" if self.label_type == 'binary' else "MIS_CELLS_PROB"
                padding_needed = num_nodes - len(full_label_array)
                print(
                    f"Warning: '{label_name}' length ({len(full_label_array)}) < expected ({num_nodes}) "
                    f"for file '{graph_path}'. Padding with zeros."
                )
                if self.label_type == 'binary':
                    full_label_array += [0] * padding_needed
                else:
                    full_label_array += [0.0] * padding_needed

            # Now slice up to num_nodes (in case it's too long)
            full_label_array = full_label_array[:num_nodes]

            # Convert to torch
            y = torch.tensor(full_label_array, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data


class GCNForMIS(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) for node-level binary or probability-based classification.
    """

    def __init__(self, hidden_channels=16, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Initialize GCN layers
        self.convs = nn.ModuleList()
        # Input layer
        self.convs.append(GCNConv(in_channels=1, out_channels=hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels))

        # Output layer
        self.out_conv = GCNConv(in_channels=hidden_channels, out_channels=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass through hidden GCN layers with ReLU activation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Output layer
        x = self.out_conv(x, edge_index)

        # Apply sigmoid for binary or probability-based classification
        return torch.sigmoid(x).squeeze(-1)  # [num_nodes]


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Also tracks the best validation MAE.
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
                print(f"Validation loss decreased. Resetting patience.")
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


def train(model, loader, optimizer, device, label_type='binary'):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # [num_nodes]

        # BCE for either binary or prob labels
        loss = F.binary_cross_entropy(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, label_type='binary'):
    """
    Evaluates the model on a given dataset.
    Returns (avg_loss, accuracy_or_mae).
    """
    model.eval()
    total_loss = 0
    correct_or_sum = 0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        total_loss += loss.item()

        if label_type == 'binary':
            # Accuracy
            preds = (out >= 0.5).float()
            correct_or_sum += (preds == data.y).sum().item()
            total_nodes += data.y.numel()
        else:
            # Probability => compute MAE
            mae_batch = torch.abs(out - data.y).sum().item()
            correct_or_sum += mae_batch
            total_nodes += data.y.numel()

    avg_loss = total_loss / len(loader)
    if label_type == 'binary':
        accuracy = correct_or_sum / total_nodes if total_nodes > 0 else 0.0
        return avg_loss, accuracy
    else:
        mae = correct_or_sum / total_nodes if total_nodes > 0 else 0.0
        return avg_loss, mae


@torch.no_grad()
def calculate_baseline_mae(loader, label_type='prob'):
    """
    Calculates the MAE for a baseline that predicts 0.5 for every node.
    """
    total_mae = 0
    total_nodes = 0

    for data in loader:
        # Baseline = 0.5
        baseline_preds = torch.full_like(data.y, 0.5)
        mae = torch.abs(baseline_preds - data.y).sum().item()
        total_mae += mae
        total_nodes += data.y.numel()

    return total_mae / total_nodes if total_nodes > 0 else 0.0
