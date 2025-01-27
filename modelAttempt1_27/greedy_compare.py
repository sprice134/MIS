import os
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, random_split

from sklearn.metrics import confusion_matrix
import pandas as pd
import networkx as nx

###############################################################################
# User-configurable settings (hardcoded instead of argparse)
###############################################################################
JSON_PATH = "all_mis_results_7.json"
EDGELIST_DIR = "../nonIsoEval/noniso_7_networkx"
NUM_NODES = 7      # The graphs have nodes labeled 0..6
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
NUM_LAYERS = 7
MODEL_SAVE_PATH = "best_model.pth"
OUTPUT_CSV = "mis_comparison_results.csv"

###############################################################################
# 1) Dataset definition
###############################################################################
class MISGraphDataset(Dataset):
    """
    A dataset that:
    - Reads a JSON file containing, for each graph, its "file_path" and "MIS_CELLS".
    - Loads the corresponding .edgelist file for each entry in the JSON.
    - Produces a PyTorch Geometric `Data` object with node feature `x`, `edge_index`,
      and label `y` = MIS_CELLS.
    """

    def __init__(self, json_path, edgelist_dir, n=7):
        super().__init__()
        self.n = n
        self.edgelist_dir = edgelist_dir

        # 1) Read the JSON. It should be a list of objects (not a dict).
        with open(json_path, 'r') as f:
            mis_data = json.load(f)  # must be a list

        # 2) Build a dictionary: file_path -> MIS_CELLS
        self.labels_dict = {}
        for entry in mis_data:
            path_key = entry["file_path"]
            self.labels_dict[path_key] = entry["MIS_CELLS"]

        # 3) Gather all .edgelist files in the directory
        all_files_in_dir = [
            os.path.join(edgelist_dir, f)
            for f in os.listdir(edgelist_dir)
            if f.endswith(".edgelist")
        ]
        # Filter to keep only those which appear in the JSON dictionary
        self.file_paths = [p for p in all_files_in_dir if p in self.labels_dict]

        # Sort for reproducibility
        self.file_paths.sort()

        if not self.file_paths:
            print("Warning: No matching .edgelist files found in JSON or directory!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        graph_path = self.file_paths[idx]

        # We'll gather edges and nodes
        edges = []
        nodes = set()

        # 1) Read each line of the .edgelist
        with open(graph_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Skip blank lines
                    continue
                tokens = line.split()
                if len(tokens) == 2:
                    # Format: "u v"
                    try:
                        u, v = int(tokens[0]), int(tokens[1])
                        edges.append((u, v))
                        edges.append((v, u))  # undirected
                        nodes.add(u)
                        nodes.add(v)
                    except ValueError:
                        # Skip lines where conversion to int fails
                        print(f"Warning: Non-integer values in line '{line}' in file {graph_path}. Skipping.")
                        continue
                elif len(tokens) == 1:
                    # Single token indicates an isolated node
                    try:
                        node_id = int(tokens[0])
                        nodes.add(node_id)
                    except ValueError:
                        # Skip lines where conversion to int fails
                        print(f"Warning: Non-integer value in line '{line}' in file {graph_path}. Skipping.")
                        continue
                else:
                    # Skip lines that don't conform
                    print(f"Warning: Unexpected format in line '{line}' in file {graph_path}. Skipping.")
                    continue

        # 2) Validate node IDs
        for nd in nodes:
            if nd < 0 or nd >= self.n:
                raise ValueError(f"Node ID {nd} is out of range [0..{self.n-1}] in file {graph_path}")

        # 3) Create the edge_index tensor
        if len(edges) == 0:
            # No edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 4) Create node features. For demonstration, use all ones [n,1].
        x = torch.ones((self.n, 1), dtype=torch.float)

        # 5) Retrieve MIS_CELLS from the dictionary. It's a list of 0/1 of length n.
        mis_cells = self.labels_dict[graph_path]
        labels = torch.tensor(mis_cells, dtype=torch.float)

        # 6) Build the PyG data object
        data = Data(x=x, edge_index=edge_index, y=labels)
        data.graph_path = graph_path  # Add graph path for reference
        return data

###############################################################################
# 2) Define a simple GCN for node-level classification
###############################################################################
class GCNForMIS(nn.Module):
    def __init__(self, hidden_channels=16, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels

        # For a simple example, we'll do GCNConv -> ReLU -> GCNConv -> ReLU -> ... -> Output
        self.convs = nn.ModuleList()
        # First layer: input = 1 feature dimension
        self.convs.append(GCNConv(in_channels=1, out_channels=hidden_channels))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels))

        # Output layer: hidden_channels -> 1
        self.out_conv = GCNConv(in_channels=hidden_channels, out_channels=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass through the hidden GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Output layer -> [num_nodes, 1]
        x = self.out_conv(x, edge_index)

        # We want to interpret these as probabilities of node=1, so apply sigmoid
        return torch.sigmoid(x).squeeze(-1)  # shape = [num_nodes]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        total_loss += loss.item()

        # threshold at 0.5
        pred = (out >= 0.5).float()
        correct += (pred == data.y).sum().item()
        total_nodes += data.y.shape[0]

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_nodes if total_nodes > 0 else 0.0
    return avg_loss, accuracy

###############################################################################
# 5) Greedy MIS Algorithm
###############################################################################
def greedy_mis_min_degree(edge_list, num_nodes):
    """
    Computes a Maximal Independent Set (MIS) using a minimum degree greedy algorithm.

    Args:
        edge_list (list of tuples): List of edges in the graph.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        set: Set of node indices included in the MIS.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)

    mis = set()
    nodes = set(G.nodes())
    while nodes:
        # Select node with minimum degree
        min_degree_node = min(nodes, key=lambda x: G.degree(x))
        mis.add(min_degree_node)
        # Remove the node and its neighbors from the graph
        neighbors = set(G.neighbors(min_degree_node))
        nodes.remove(min_degree_node)
        nodes -= neighbors
        G.remove_node(min_degree_node)
        G.remove_nodes_from(neighbors)
    return mis

###############################################################################
# 6) Main script
###############################################################################
def main():
    # Seed for reproducibility
    torch.manual_seed(42)

    # 1) Create dataset
    dataset = MISGraphDataset(
        json_path=JSON_PATH,
        edgelist_dir=EDGELIST_DIR,
        n=NUM_NODES
    )

    if len(dataset) == 0:
        print("No graphs found. Exiting.")
        return

    # 2) Split into test set (100%)
    test_dataset = dataset

    print(f"Test Dataset Size: {len(test_dataset)}")

    # 3) Create test data loader
    test_loader  = GeometricDataLoader(test_dataset,  batch_size=1, shuffle=False)  # Batch size 1 for per-graph processing

    # 4) Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS
    ).to(device)

    # 5) Load the best model
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded the best model from {MODEL_SAVE_PATH}.")
    else:
        print(f"Warning: {MODEL_SAVE_PATH} not found. Exiting.")
        return

    # 6) Evaluate on Test Set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 7) Inference and MIS Comparison on Test Set
    print("\nPerforming MIS comparison on the test set...")

    model.eval()
    results = []
    same_count = 0
    vanilla_better = 0
    augmented_better = 0

    for i, data in enumerate(test_loader):
        data = data.to(device)
        graph_path = data.graph_path[0]  # Assuming single graph per batch

        # Extract edge list
        edge_index = data.edge_index.cpu().numpy()
        edge_list = list(zip(edge_index[0], edge_index[1]))
        # Remove duplicate edges since it's undirected
        edge_set = set()
        for u, v in edge_list:
            if u < v:
                edge_set.add((u, v))
        unique_edge_list = list(edge_set)

        # Compute vanilla greedy MIS
        vanilla_mis = greedy_mis_min_degree(unique_edge_list, NUM_NODES)
        vanilla_mis_size = len(vanilla_mis)

        # Model predictions
        with torch.no_grad():
            out = model(data)  # shape [num_nodes]
            preds = (out >= 0.5).float().cpu().numpy()
        preds = preds.astype(int)

        # Nodes to delete (predicted as 0)
        nodes_to_delete = [node for node, pred in enumerate(preds) if pred == 0]

        # Create augmented edge list by removing nodes_to_delete
        augmented_nodes = set(range(NUM_NODES)) - set(nodes_to_delete)
        augmented_edge_list = [(u, v) for u, v in unique_edge_list if u in augmented_nodes and v in augmented_nodes]

        # Compute augmented greedy MIS
        augmented_mis = greedy_mis_min_degree(augmented_edge_list, len(augmented_nodes))
        augmented_mis_size = len(augmented_mis)

        # Save results
        graph_name = os.path.basename(graph_path)
        results.append({
            "graph_name": graph_name,
            "vanilla_greedy_mis_size": vanilla_mis_size,
            "augmented_greedy_mis_size": augmented_mis_size
        })

        # Compare MIS sizes
        if vanilla_mis_size == augmented_mis_size:
            same_count += 1
        elif vanilla_mis_size > augmented_mis_size:
            vanilla_better += 1
        else:
            augmented_better += 1

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}.")

    # Print summary
    print("\nSummary of MIS Comparison:")
    print(f"Number of graphs with the same MIS size: {same_count}")
    print(f"Number of graphs where vanilla greedy MIS was better: {vanilla_better}")
    print(f"Number of graphs where augmented greedy MIS was better: {augmented_better}")

if __name__ == "__main__":
    main()
