#!/usr/bin/env python3
import os
import torch
import sys
sys.path.append('modelAttempt2_5')
from tools3 import MISGraphDataset, GCNForMIS  # Ensure these are accessible

def print_model_predictions(graph_filepath, model_filepath):
    # Load the graph using MISGraphDataset.
    dataset = MISGraphDataset(
        json_paths=[],                # no JSON info needed for this minimal example
        edgelist_dirs=[graph_filepath],
        label_type='prob'
    )
    if len(dataset) == 0:
        print("Error: Could not load any graphs from the provided file.")
        return

    graph_data = dataset[0]
    num_nodes = graph_data.num_nodes
    print(f"Loaded graph with {num_nodes} nodes from '{graph_filepath}'.")

    # Set up device and load the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Here we use default parameters; adjust hidden_channels/num_layers as needed.
    model = GCNForMIS(hidden_channels=176, num_layers=28).to(device)
    if not os.path.exists(model_filepath):
        print(f"Error: Model file '{model_filepath}' not found.")
        return
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model.eval()

    # Move graph data to device and run model prediction.
    graph_data = graph_data.to(device)
    with torch.no_grad():
        preds = model(graph_data)
    # Convert predictions to a numpy array and flatten.
    preds_np = preds.cpu().numpy().flatten()

    # Print predictions for each node.
    print("Model predictions per node:")
    for i, pred in enumerate(preds_np):
        print(f"  Node {i}: {pred:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Print model predictions for each node of a graph"
    )
    parser.add_argument("--graph_filepath", type=str, required=True,
                        help="Filepath to the graph edgelist (directory containing the edgelist file).")
    parser.add_argument("--model_filepath", type=str, required=True,
                        help="Filepath to the model weights (e.g., best_model_prob.pth).")
    args = parser.parse_args()
    print_model_predictions(args.graph_filepath, args.model_filepath)
    '''
    python singleInference.py \
        --graph_filepath test_generated_graphs/nodes_10/removal_75percent/graph_11.edgelist \
        --model_filepath best_model_prob_32_176_28_0.001_v5.pth

    '''
