# main.py

import os
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader

from tools import MISGraphDataset, GCNForMIS, EarlyStopping, train, evaluate
from sklearn.metrics import confusion_matrix

###############################################################################
# User-configurable settings
###############################################################################
JSON_PATHS = [
    "all_mis_results_3.json",
    "all_mis_results_4.json",
    "all_mis_results_5.json",
    "all_mis_results_6.json",
    "all_mis_results_7.json",
]

EDGELIST_DIRS = [
    "../nonIsoEval/noniso_3_networkx",
    "../nonIsoEval/noniso_4_networkx",
    "../nonIsoEval/noniso_5_networkx",
    "../nonIsoEval/noniso_6_networkx",
    "../nonIsoEval/noniso_7_networkx",
]

BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
NUM_LAYERS = 7
LEARNING_RATE = 1e-3
EPOCHS = 1000
PATIENCE = 20        # For Early Stopping
MODEL_SAVE_PATH = "best_model.pth"

###############################################################################
# 5) Main script
###############################################################################
def main():
    # Seed for reproducibility
    torch.manual_seed(42)

    # 1) Create ONE combined dataset for all N=3..7
    dataset = MISGraphDataset(
        json_paths=JSON_PATHS,
        edgelist_dirs=EDGELIST_DIRS
    )

    if len(dataset) == 0:
        print("No graphs found. Exiting.")
        return

    # 2) Split into train (70%), validation (20%), and test (10%) sets
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

    # 3) Create data loaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = GeometricDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = GeometricDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5) Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=MODEL_SAVE_PATH)

    # 6) Training loop with Early Stopping
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
        out = model(data)  # Shape: [num_nodes]
        preds = (out >= 0.5).float().cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        print(f"Graph batch {i}: predicted (sigmoid) = {preds}, labels = {labels}")

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
