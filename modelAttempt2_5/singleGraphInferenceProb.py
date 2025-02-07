# singleGraphInferenceProb.py

#!/usr/bin/env python3

import torch
from tools3 import GCNForMIS, MISGraphDataset
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.colors as mcolors  # Importing colors for hex conversion

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on a single graph using a trained GCN model."
    )
    parser.add_argument(
        "--edgelist_path",
        type=str,
        required=True,
        help="Path to the edgelist file containing the graph data."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_model_prob.pth",
        help="Path to the saved trained model."
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Number of hidden channels in the GCN."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of GCN layers."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Optional path to the JSON file containing MIS labels."
    )
    args = parser.parse_args()

    edgelist_path = args.edgelist_path
    model_save_path = args.model_save_path
    hidden_channels = args.hidden_channels
    num_layers = args.num_layers
    json_path = args.json_path

    # Ensure edgelist and model files exist
    if not os.path.exists(edgelist_path):
        print(f"Error: Edgelist file '{edgelist_path}' does not exist.")
        return

    if not os.path.exists(model_save_path):
        print(f"Error: Model file '{model_save_path}' does not exist.")
        return

    # If JSON path is provided, ensure it exists
    if json_path and not os.path.exists(json_path):
        print(f"Error: JSON file '{json_path}' does not exist.")
        return

    # Load the dataset for inference
    dataset = MISGraphDataset(
        json_paths=[],  # Empty list since we're not using JSON files for dataset
        edgelist_dirs=[edgelist_path],  # Load directly from the edgelist
        label_type='prob'  # Set to 'prob' or 'binary' as needed
    )

    if len(dataset) == 0:
        print("Error: Dataset is empty. Exiting.")
        return

    print(f"Loaded graph with {dataset[0].num_nodes} nodes.")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNForMIS(
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    print(f"Model loaded from '{model_save_path}'.")

    # Get the graph data
    graph_data = dataset[0].to(device)

    # Predict node scores
    with torch.no_grad():
        scores = model(graph_data).cpu().numpy()

    # Print scores for each node
    print("Node scores:")
    for i, score in enumerate(scores):
        print(f"Node {i}: {score:.4f}")

    # ----- Augmentation: Plotting the Graph with Predicted Scores -----

    # Create the temp_images/ directory if it doesn't exist
    images_dir = "temp_images"
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, "graph.png")

    # Convert edge_index to a list of edges
    edge_index = graph_data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Add nodes to ensure all nodes are present (including isolated nodes)
    G.add_nodes_from(range(graph_data.num_nodes))

    # Define positions for a consistent layout
    pos = nx.spring_layout(G, seed=42)  # You can choose other layouts like kamada_kawai_layout

    # Initialize node colors to default (lightblue)
    node_colors = ['lightblue'] * graph_data.num_nodes

    # If JSON path is provided, update node colors based on MIS_CELLS_PROB
    if json_path:
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Search for the matching file_path
            matching_entry = None
            for entry in json_data:
                # Normalize paths to handle different path formats
                entry_file_path = os.path.normpath(entry.get("file_path", ""))
                target_file_path = os.path.normpath(edgelist_path)
                if os.path.basename(entry_file_path) == os.path.basename(target_file_path):
                    # Optional: Further path matching can be done here if needed
                    matching_entry = entry
                    break

            if matching_entry:
                mis_cells_prob = matching_entry.get("MIS_CELLS_PROB", [])
                print(f"MIS_CELLS_PROB: {mis_cells_prob}")
                if len(mis_cells_prob) != graph_data.num_nodes:
                    print(f"Warning: Length of MIS_CELLS_PROB ({len(mis_cells_prob)}) does not match number of nodes ({graph_data.num_nodes}).")
                else:
                    # Update node colors based on MIS_CELLS_PROB using a colormap
                    norm = plt.Normalize(vmin=min(mis_cells_prob), vmax=max(mis_cells_prob))
                    cmap = plt.cm.viridis  # Choose a colormap
                    node_colors = [mcolors.to_hex(cmap(norm(prob))) for prob in mis_cells_prob]  # Convert to hex
                    print(f"Node colors: {node_colors}")
                    print("Node colors updated based on MIS_CELLS_PROB from JSON.")
            else:
                print(f"Warning: No matching entry found in JSON for edgelist path '{edgelist_path}'. Using default node colors.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file '{json_path}': {e}")
            print("Proceeding with default node colors.")
        except Exception as e:
            print(f"Unexpected error while processing JSON file: {e}")
            print("Proceeding with default node colors.")

    # Create labels for node IDs
    node_labels = {i: str(i) for i in G.nodes()}

    # Align node_colors with NetworkX's node order
    node_order = list(G.nodes())  # Retrieve nodes in NetworkX's internal order
    node_colors_ordered = [node_colors[node] for node in node_order]

    # Verify alignment
    print(f"Node order in NetworkX: {node_order}")
    print(f"Corresponding node colors: {node_colors_ordered}")

    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Draw nodes with updated colors
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors_ordered)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw node labels (node IDs)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

    # Add score labels below each node
    # Calculate an offset for the labels
    # Determine the scale of the plot to set an appropriate offset
    x_vals, y_vals = zip(*pos.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    offset = (y_max - y_min) * 0.05  # 5% of the y-axis range

    for node, (x, y) in pos.items():
        score = scores[node]
        plt.text(x, y - offset, f"{score:.4f}", fontsize=8, ha='center', va='top', color='red')

    # Add a color bar for MIS_CELLS_PROB
    if json_path and matching_entry and len(mis_cells_prob) == graph_data.num_nodes:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.5)
        cbar.set_label('MIS_CELLS_PROB')
    else:
        # If MIS_CELLS_PROB is not available, use predicted scores for color bar
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else np.zeros_like(scores)
        cmap_scores = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap_scores, norm=plt.Normalize(vmin=scores.min(), vmax=scores.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.5)
        cbar.set_label('Predicted Score')

    plt.title("Graph Visualization with MIS_CELLS_PROB and Predicted Scores")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_path, format="PNG")
    plt.close()

    print(f"Graph image saved to '{image_path}'.")

if __name__ == "__main__":
    main()

    '''
    Example usage with JSON:
    python singleGraphInferenceProb.py \
    --edgelist_path generated_graphs/nodes_10/removal_30percent/graph_1.edgelist \
    --model_save_path best_model_prob.pth \
    --json_path /home/sprice/MIS/modelAttempt1_28/mis_results_grouped_v2/nodes_10_removal_30percent.json

    Example usage without JSON:
    python singleGraphInferenceProb.py \
    --edgelist_path generated_graphs/nodes_10/removal_85percent/graph_17.edgelist \
    --model_save_path best_model_prob.pth \
    --json_path /home/sprice/MIS/modelAttempt1_28/mis_results_grouped_v2/nodes_10_removal_85percent.json


    python singleGraphInferenceProb.py \
    --edgelist_path generated_graphs/nodes_10/removal_85percent/graph_13.edgelist \
    --model_save_path best_model_prob.pth \
    --json_path /home/sprice/MIS/modelAttempt1_28/mis_results_grouped_v2/nodes_10_removal_85percent.json

    python singleGraphInferenceProb.py \
    --edgelist_path generated_graphs/nodes_15/removal_85percent/graph_8.edgelist \
    --model_save_path best_model_prob.pth \
    --json_path /home/sprice/MIS/modelAttempt1_28/mis_results_grouped_v2/nodes_15_removal_85percent.json

    '''
