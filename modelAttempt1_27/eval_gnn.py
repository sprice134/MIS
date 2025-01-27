import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, random_split

from sklearn.metrics import confusion_matrix

###############################################################################
# User-configurable settings (hardcoded instead of argparse)
###############################################################################
JSON_PATH = "all_mis_results_7.json"
EDGELIST_DIR = "../nonIsoEval/noniso_7_networkx"
NUM_NODES = 7      # The graphs have nodes labeled 0..6
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
NUM_LAYERS = 7
LEARNING_RATE = 1e-3
EPOCHS = 1000
PATIENCE = 20        # For Early Stopping
MODEL_SAVE_PATH = "best_model.pth"

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

###############################################################################
# 3) Early Stopping Class
###############################################################################
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
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
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
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

###############################################################################
# 4) Training and evaluation routines
###############################################################################
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # shape [num_nodes]

        # BCE since each node is 0/1
        # 'out' is already sigmoid-ed
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

        # threshold at 0.5
        pred = (out >= 0.5).float()
        correct += (pred == data.y).sum().item()
        total_nodes += data.y.shape[0]

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_nodes if total_nodes > 0 else 0.0
    return avg_loss, accuracy

###############################################################################
# 5) Main script (no argparse)
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

    # 2) Split into train (70%), validation (20%), and test (10%) sets
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

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

    # 3) Create data loaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = GeometricDataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # 4) Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5) Initialize Early Stopping
    '''early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=MODEL_SAVE_PATH)

    # 6) Training loop with Early Stopping
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | "
              f"Validation Acc: {val_acc:.4f}")

        # Early Stopping needs the validation loss to check if it has decreased
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break'''

    # 7) Load the best model
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded the best model from {MODEL_SAVE_PATH}.")
    else:
        print(f"Warning: {MODEL_SAVE_PATH} not found. Using the current model.")

    # 8) Evaluate on Test Set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 9) Example inference on test set with Confusion Matrix
    print("\nInference on the test set and Confusion Matrix:")
    model.eval()
    all_preds = []
    all_labels = []
    for i, data in enumerate(test_loader):
        data = data.to(device)
        out = model(data)  # shape [num_nodes]
        preds = (out >= 0.5).float().cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        # print(f"Graph batch {i}: predicted (sigmoid) = {preds}, "
            #   f"labels = {labels}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nConfusion Matrix Breakdown:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

if __name__ == "__main__":
    main()
