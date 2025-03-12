#!/usr/bin/env python3
import os
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tools4 import MISGraphDataset, GCNForMIS, EarlyStopping, calculate_baseline_mae
from sklearn.metrics import confusion_matrix
import argparse
import random
import numpy as np
import torch.nn.functional as F

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # For binary mode, expects raw logits if using BCEWithLogitsLoss.
        loss = loss_fn(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate_epoch(model, loader, device, loss_fn, label_type):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y.float())
            total_loss += loss.item() * data.num_graphs
            if label_type == 'binary':
                preds = torch.sigmoid(out)
            else:
                preds = out
            total_mae += torch.abs(preds - data.y.float()).sum().item()
            total_nodes += data.y.numel()
    mae = total_mae / total_nodes
    return total_loss / len(loader.dataset), mae


def main():
    parser = argparse.ArgumentParser(
        description="Train a GCN model on Maximum Independent Set (MIS) data with configurable label type."
    )
    parser.add_argument("--node_counts", type=int, nargs='+', default=list(range(10, 55, 5)),
                        help="List of node counts to include, e.g., --node_counts 10 15 20 ... 50")
    parser.add_argument("--removal_percents", type=int, nargs='+', default=list(range(15, 90, 5)),
                        help="List of edge removal percentages to include, e.g., --removal_percents 15 20 25 ... 85")
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped",
                        help="Directory where MIS JSON result files are stored.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Number of hidden channels in the GCN.")
    parser.add_argument("--num_layers", type=int, default=7, help="Number of GCN layers.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")
    parser.add_argument("--model_save_path", type=str, default="best_model_binary.pth",
                        help="Path to save the best model.")
    parser.add_argument("--label_type", type=str, default="binary", choices=["binary", "prob"],
                        help="Label type: 'binary' or 'prob'")
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
    label_type = args.label_type.lower()

    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(output_dir, node_counts, removal_percents)
    if not json_paths or not edgelist_dirs:
        print("Error: No valid JSON files or edgelist directories found. Exiting.")
        return
    print(f"Total JSON files found: {len(json_paths)}")
    print(f"Total edgelist directories found: {len(edgelist_dirs)}")

    dataset = MISGraphDataset(json_paths=json_paths, edgelist_dirs=edgelist_dirs, label_type=label_type)
    if len(dataset) == 0:
        print("Error: Dataset is empty after loading. Exiting.")
        return
    print(f"Total graphs in dataset: {len(dataset)}")

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val],
                                               generator=torch.Generator().manual_seed(42))
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")

    train_loader = GeometricDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nCalculating baseline MAE (predicting 0.5 for every node):")
    baseline_train_mae = calculate_baseline_mae(train_loader, label_type=label_type)
    baseline_val_mae = calculate_baseline_mae(val_loader, label_type=label_type)
    print(f"Baseline Training MAE: {baseline_train_mae:.4f}")
    print(f"Baseline Validation MAE: {baseline_val_mae:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if label_type == 'binary':
        # In binary mode, do not apply sigmoid in the model; use BCEWithLogitsLoss.
        model = GCNForMIS(hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS, apply_sigmoid=False).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        # In probability mode, apply sigmoid in the model and use BCELoss.
        model = GCNForMIS(hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS, apply_sigmoid=True).to(device)
        loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=MODEL_SAVE_PATH)

    best_validation_mae = None
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_mae = evaluate_epoch(model, val_loader, device, loss_fn, label_type)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation MAE: {val_mae:.4f}")
        early_stopping(val_loss, val_mae, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            best_validation_mae = early_stopping.best_mae
            break

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded the best model from '{MODEL_SAVE_PATH}'.")
    else:
        print(f"Warning: '{MODEL_SAVE_PATH}' not found. Using the current model.")

    test_loss, test_metric = evaluate_epoch(model, val_loader, device, loss_fn, label_type)
    metric_name = "Accuracy" if label_type == 'binary' else "MAE"
    print(f"\nTest Loss: {test_loss:.4f} | Test {metric_name}: {test_metric:.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            if label_type == 'binary':
                preds = torch.sigmoid(out)
            else:
                preds = out
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    thresholds = [best_validation_mae if best_validation_mae is not None else 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("\nConfusion Matrix and F1 Score for Different Thresholds:")
    for thresh in thresholds:
        binarized_preds = [1 if pred >= thresh else 0 for pred in all_preds]
        binarized_labels = [1 if label >= 0.5 else 0 for label in all_labels]
        cm = confusion_matrix(binarized_labels, binarized_preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            precision = recall = f1 = 0.0
        print(f"\nThreshold: {thresh:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
    '''
    python modelTrain_binary.py \
        --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70\
        --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
        --output_dir mis_results_grouped_v3 \
        --batch_size 32\
        --hidden_channels 176 \
        --num_layers 28 \
        --learning_rate 0.015 \
        --epochs 1000 \
        --patience 35 \
        --model_save_path best_model_binary_32_176_28_0.001_v1.pth
    '''