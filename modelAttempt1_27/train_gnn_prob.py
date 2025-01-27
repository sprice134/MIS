import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, random_split

###############################################################################
# User-configurable settings (hardcoded instead of argparse)
###############################################################################
JSON_PATH = "all_mis_results_7.json"
EDGELIST_DIR = "../nonIsoEval/noniso_7_networkx"
NUM_NODES = 7      # The graphs have nodes labeled 0..6
BATCH_SIZE = 16
HIDDEN_CHANNELS = 64
NUM_LAYERS = 6
LEARNING_RATE = 1e-3
EPOCHS = 250


###############################################################################
# 1) Dataset definition
###############################################################################
class MISGraphDataset(Dataset):
    """
    A dataset that:
    - Reads a JSON file containing, for each graph, its "file_path" and "MIS_CELLS_PROB".
    - Loads the corresponding .edgelist file for each entry in the JSON.
    - Produces a PyTorch Geometric `Data` object with node feature `x`, `edge_index`,
      and label `y` = MIS_CELLS_PROB (soft labels).
    """

    def __init__(self, json_path, edgelist_dir, n=7):
        super().__init__()
        self.n = n
        self.edgelist_dir = edgelist_dir

        # 1) Read the JSON.  It should be a list of objects.
        with open(json_path, 'r') as f:
            mis_data = json.load(f)  # must be a list

        # 2) Build a dictionary: file_path -> MIS_CELLS_PROB
        self.labels_dict = {}
        for entry in mis_data:
            path_key = entry["file_path"]
            # We take MIS_CELLS_PROB (e.g. [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            self.labels_dict[path_key] = entry["MIS_CELLS_PROB"]

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
                    u, v = int(tokens[0]), int(tokens[1])
                    edges.append((u, v))
                    edges.append((v, u))  # undirected
                    nodes.add(u)
                    nodes.add(v)
                elif len(tokens) == 1:
                    # Single token indicates an isolated node
                    node_id = int(tokens[0])
                    nodes.add(node_id)
                else:
                    # Skip lines that don't conform
                    continue

        for nd in nodes:
            if nd < 0 or nd >= self.n:
                raise ValueError(f"Node ID {nd} is out of range [0..{self.n-1}] in file {graph_path}")

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        x = torch.ones((self.n, 1), dtype=torch.float)

        # Instead of MIS_CELLS (0 or 1), we read MIS_CELLS_PROB (a float in [0,1])
        mis_cells_prob = self.labels_dict[graph_path]
        labels = torch.tensor(mis_cells_prob, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=labels)
        return data


###############################################################################
# 2) Define a simple GCN for node-level classification (soft labels)
###############################################################################
class GCNForMIS(nn.Module):
    def __init__(self, hidden_channels=16, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.convs = nn.ModuleList()
        # First layer: input = 1 feature dimension
        self.convs.append(GCNConv(in_channels=1, out_channels=hidden_channels))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels))

        # Output layer: hidden_channels -> 1
        self.out_conv = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.out_conv(x, edge_index)
        # We still apply sigmoid to interpret outputs as a probability
        return torch.sigmoid(x).squeeze(-1)  # shape = [num_nodes]


###############################################################################
# 3) Training and evaluation routines
###############################################################################
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # shape [num_nodes]

        # 'out' is in [0,1], mis_cells_prob in [0,1].
        # We'll use BCE with soft labels:
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

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

        # We'll threshold at 0.5 to see how often we match "prob >= 0.5" vs. actual labels > 0.5
        pred = (out >= 0.5).float()
        true_binary = (data.y >= 0.5).float()  # Convert the "soft" label into a binary label for accuracy
        correct += (pred == true_binary).sum().item()
        total_nodes += data.y.numel()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_nodes if total_nodes > 0 else 0.0
    return avg_loss, accuracy


###############################################################################
# 4) Main script (no argparse)
###############################################################################
def main():
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

    # 2) Split into train/test sets (80/20)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size:  {len(test_dataset)}")

    # 3) Create data loaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = GeometricDataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # 4) Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5) Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")

    # 6) Example inference on test set
    print("\nInference on the test set:")
    model.eval()
    for i, data in enumerate(test_loader):
        data = data.to(device)
        out = model(data)  # shape [num_nodes]
        print(f"Graph batch {i}: predicted (sigmoid) = {out.data.cpu().numpy()}, "
              f"labels (prob) = {data.y.cpu().numpy()}")

if __name__ == "__main__":
    main()
