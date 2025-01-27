# tools.py

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
        - x: node features (all ones here) [num_nodes, 1]
        - edge_index: shape [2, num_edges], undirected
        - y: labels [num_nodes] (0/1 per node or probability between 0 and 1)
    """

    def __init__(self, json_paths, edgelist_dirs, label_type='binary'):
        """
        Initializes the dataset.

        Args:
            json_paths (list of str): List of JSON file paths containing MIS results.
            edgelist_dirs (list of str): List of directories containing .edgelist files.
            label_type (str): Type of labels to use. Options are 'binary' or 'prob'.
        """
        super().__init__()

        # Ensure the lists are of the same length
        assert len(json_paths) == len(edgelist_dirs), \
            "JSON_PATHS and EDGELIST_DIRS must have the same length."

        self.file_paths = []
        self.labels_dict = {}
        self.label_type = label_type.lower()

        if self.label_type not in ['binary', 'prob']:
            raise ValueError("label_type must be either 'binary' or 'prob'.")

        # Iterate over each (json_path, edgelist_dir) pair
        for json_path, edgelist_dir in zip(json_paths, edgelist_dirs):
            # Load the JSON file
            if not os.path.exists(json_path):
                print(f"Warning: JSON file '{json_path}' does not exist. Skipping.")
                continue

            with open(json_path, 'r') as f:
                try:
                    mis_data = json.load(f)  # Expecting a list of dictionaries
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file '{json_path}': {e}")
                    continue

            # Process each entry in the JSON
            for entry in mis_data:
                base_name = os.path.basename(entry.get("file_path", ""))
                if not base_name:
                    print(f"Warning: Missing 'file_path' in entry {entry}. Skipping.")
                    continue

                full_edgelist_path = os.path.join(edgelist_dir, base_name)
                if self.label_type == 'binary':
                    label_key = "MIS_CELLS"
                else:
                    label_key = "MIS_CELLS_PROB"

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

        # Read the .edgelist file
        with open(graph_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                tokens = line.split()

                if len(tokens) == 2:
                    # Edge definition "u v"
                    try:
                        u, v = int(tokens[0]), int(tokens[1])
                        edges.append((u, v))
                        edges.append((v, u))  # Assuming undirected graphs
                        node_ids.update([u, v])
                    except ValueError:
                        print(f"Warning: Non-integer values in line '{line}' in file '{graph_path}'. Skipping.")
                        continue
                elif len(tokens) == 1:
                    # Isolated node "u"
                    try:
                        u = int(tokens[0])
                        node_ids.add(u)
                    except ValueError:
                        print(f"Warning: Non-integer value in line '{line}' in file '{graph_path}'. Skipping.")
                        continue
                else:
                    print(f"Warning: Unexpected format in line '{line}' in file '{graph_path}'. Skipping.")
                    continue

        # Reindex node IDs to [0, num_nodes - 1]
        unique_nodes = sorted(node_ids)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
        remapped_edges = [(old_to_new[u], old_to_new[v]) for u, v in edges]
        num_nodes = len(unique_nodes)

        # Create edge_index tensor
        if remapped_edges:
            edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Node features (all ones)
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        # Retrieve and reindex labels
        full_label_array = self.labels_dict.get(graph_path, [])
        if self.label_type == 'binary':
            expected_length = max(unique_nodes) + 1 if unique_nodes else 0
            if len(full_label_array) < expected_length:
                print(f"Warning: 'MIS_CELLS' length for '{graph_path}' is shorter than expected. Padding with zeros.")
                full_label_array += [0] * (expected_length - len(full_label_array))

            try:
                y_list = [full_label_array[old_id] for old_id in unique_nodes]
            except IndexError as e:
                print(f"Error: {e} when accessing 'MIS_CELLS' for '{graph_path}'. Assigning zeros.")
                y_list = [0] * num_nodes
        else:  # label_type == 'prob'
            expected_length = max(unique_nodes) + 1 if unique_nodes else 0
            if len(full_label_array) < expected_length:
                print(f"Warning: 'MIS_CELLS_PROB' length for '{graph_path}' is shorter than expected. Padding with zeros.")
                full_label_array += [0.0] * (expected_length - len(full_label_array))

            try:
                y_list = [float(full_label_array[old_id]) for old_id in unique_nodes]
            except (IndexError, ValueError) as e:
                print(f"Error: {e} when accessing 'MIS_CELLS_PROB' for '{graph_path}'. Assigning zeros.")
                y_list = [0.0] * num_nodes

        y = torch.tensor(y_list, dtype=torch.float)

        # Create PyG Data object
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
        return torch.sigmoid(x).squeeze(-1)  # Shape: [num_nodes]


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Also tracks the best validation MAE.
    """

    def __init__(self, patience=20, verbose=False, delta=0.0, path='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.best_mae = None  # Track best validation MAE
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
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased to {val_loss:.4f}. Saving model to {self.path}.")


def train(model, loader, optimizer, device, label_type='binary'):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The GCN model.
        loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): Device to run the training on.
        label_type (str): Type of labels ('binary' or 'prob').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # Shape: [num_nodes]

        if label_type == 'binary':
            # Binary Cross Entropy Loss for binary labels
            loss = F.binary_cross_entropy(out, data.y)
        else:
            # Binary Cross Entropy Loss for probability labels
            # Since y is a probability, BCE can still be used
            loss = F.binary_cross_entropy(out, data.y)
            # Alternatively, Mean Squared Error can be used:
            # loss = F.mse_loss(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, label_type='binary'):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The GCN model.
        loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): Device to run the evaluation on.
        label_type (str): Type of labels ('binary' or 'prob').

    Returns:
        tuple: Average loss and accuracy (for binary) or other metrics.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        if label_type == 'binary':
            loss = F.binary_cross_entropy(out, data.y)
        else:
            loss = F.binary_cross_entropy(out, data.y)  # Or use MSE

        total_loss += loss.item()

        if label_type == 'binary':
            # Binary predictions with threshold 0.5
            preds = (out >= 0.5).float()
            correct += (preds == data.y).sum().item()
            total_nodes += data.y.shape[0]
        else:
            # For probability labels, define a suitable metric, e.g., MSE or correlation
            # Here, we'll compute Mean Absolute Error (MAE) as an example
            preds = out
            correct += torch.sum(torch.abs(preds - data.y)).item()
            total_nodes += data.y.shape[0]

    avg_loss = total_loss / len(loader)
    if label_type == 'binary':
        accuracy = correct / total_nodes if total_nodes > 0 else 0.0
        return avg_loss, accuracy
    else:
        mae = correct / total_nodes if total_nodes > 0 else 0.0
        return avg_loss, mae


@torch.no_grad()
def calculate_baseline_mae(loader, label_type='prob'):
    """
    Calculates the MAE for a baseline that predicts 0.5 for every node.
    
    Args:
        loader (DataLoader): DataLoader for the dataset (train, validation, or test).
        label_type (str): Type of labels ('binary' or 'prob'). Default is 'prob'.

    Returns:
        float: Mean Absolute Error (MAE) of the baseline predictions.
    """
    total_mae = 0
    total_nodes = 0

    for data in loader:
        # Baseline prediction is 0.5 for all nodes
        baseline_preds = torch.full_like(data.y, 0.5)  # Same shape as true labels, filled with 0.5

        # Compute MAE
        mae = torch.abs(baseline_preds - data.y).sum().item()
        total_mae += mae
        total_nodes += data.y.shape[0]

    # Return the average MAE across all nodes
    return total_mae / total_nodes if total_nodes > 0 else 0.0