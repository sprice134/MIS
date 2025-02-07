#!/usr/bin/env python3
"""
This script reads a CSV file containing MIS information and greedy approximations,
computes the RMSE (Root Mean Square Error) between the true MIS sizes and the
predicted MIS sizes from each of the four greedy metrics:
    - mis_min_degree
    - mis_random
    - mis_prob
    - mis_low_neighbor_prob

It then groups the results by number of nodes and removal percentage and produces a heatmap
for each metric. The X axis represents the number of nodes, and the Y axis represents the removal percentage.
The heatmap image is saved to disk and also displayed.

Usage:
    python analyzeGreedyData.py --csv_in mis_greedy_results.csv --img_out rmse_heatmaps.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rmse(y_true, y_pred):
    """Compute the Root Mean Square Error between y_true and y_pred."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSE for each greedy MIS approximation and plot heatmaps."
    )
    parser.add_argument("--csv_in", type=str, default="mis_greedy_results.csv",
                        help="Path to the CSV file containing MIS results.")
    parser.add_argument("--img_out", type=str, default="rmse_heatmaps.png",
                        help="Filename for the output heatmap image.")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv_in)
    except Exception as e:
        print(f"Error reading CSV file {args.csv_in}: {e}")
        return

    # Check that the expected columns are present.
    required_columns = ["true_mis", "mis_min_degree", "mis_random", "mis_prob", "mis_low_neighbor_prob",
                          "num_nodes", "removal_percent"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing from the CSV: {missing_columns}")
        return

    true_mis = df["true_mis"].values
    metrics = {
        "mis_min_degree": df["mis_min_degree"].values,
        "mis_random": df["mis_random"].values,
        "mis_prob": df["mis_prob"].values,
        "mis_low_neighbor_prob": df["mis_low_neighbor_prob"].values,
    }
    
    overall_rmse = {}
    print("Overall RMSE for each greedy MIS approximation:")
    for metric_name, predictions in metrics.items():
        error = rmse(true_mis, predictions)
        overall_rmse[metric_name] = error
        print(f"  {metric_name}: {error:.4f}")
    
    # Prepare heatmap data by grouping by removal_percent (Y axis) and num_nodes (X axis)
    # We'll compute the RMSE for each group for each metric.
    metrics_heatmaps = {}
    for metric in metrics.keys():
        grouped = df.groupby(["removal_percent", "num_nodes"]).apply(
            lambda g: rmse(g["true_mis"], g[metric])
        )
        # Unstack to form a pivot table where columns are num_nodes and index is removal_percent.
        pivot = grouped.unstack(level="num_nodes")
        metrics_heatmaps[metric] = pivot

    # Determine global vmin and vmax across all heatmaps for standardizing the color scale.
    all_values = np.concatenate([pivot.values.flatten() for pivot in metrics_heatmaps.values()])
    # Remove any NaN values which might exist if some groups are missing.
    all_values = all_values[~np.isnan(all_values)]
    global_vmin = all_values.min() if len(all_values) > 0 else 0
    global_vmax = all_values.max() if len(all_values) > 0 else 1

    # Create a 2x2 subplot for the four metrics.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RMSE Heatmaps for Greedy MIS Approximations", fontsize=16)

    subplot_mapping = {
        "mis_min_degree": (0, 0),
        "mis_random": (0, 1),
        "mis_prob": (1, 0),
        "mis_low_neighbor_prob": (1, 1)
    }

    for metric, pivot in metrics_heatmaps.items():
        ax = axes[subplot_mapping[metric]]
        # Plot the heatmap with standardized color scale.
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="viridis",
                    vmin=global_vmin, vmax=global_vmax)
        ax.set_title(f"{metric} (Overall RMSE: {overall_rmse[metric]:.2f})")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Removal Percentage")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the figure to the specified file.
    plt.savefig(args.img_out)
    print(f"Heatmap image saved to {args.img_out}")
    plt.show()

if __name__ == "__main__":
    main()
