import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# For reproducibility (optional)
np.random.seed(0)

# Generate a list of 25 random colors.
random_colors = [tuple(np.random.rand(3,)) for _ in range(25)]

# Find all CSV files with the specified prefix.
csv_files = glob.glob("bipartite_experiment_results_epislons_*.csv")

# Load and concatenate all CSV files.
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# --------------------- Function to create line chart figure ---------------------
def create_line_chart_figure(df, y_value, filename, y_label_title, coefficients_subset=None):
    """
    Generates a line chart figure for the given y_value column.
    Each subplot corresponds to a unique "Nodes" value, with one line per "Coefficient".
    An extra subplot displays a legend using a predefined random color list.
    
    Parameters:
      df: pandas DataFrame containing the data.
      y_value: column name in the DataFrame to plot on the y-axis.
      filename: the filename to save the figure.
      y_label_title: label for the y-axis.
      coefficients_subset: (optional) list of coefficient values to plot. If None, all coefficients are plotted.
    """
    # Group by Epsilon, Nodes, and Coefficient and compute the mean of the specified y_value.
    grouped = df.groupby(["Epsilon", "Nodes", "Coefficient"], as_index=False)[y_value].mean()

    # Get unique Nodes values (for subplots) and unique Coefficient values.
    nodes_values = sorted(grouped["Nodes"].unique())
    coefficients = sorted(grouped["Coefficient"].unique())
    
    # Filter coefficients if a subset is provided.
    if coefficients_subset is not None:
        coefficients = [coeff for coeff in coefficients if coeff in coefficients_subset]

    # Create an extra subplot for the legend.
    total_plots = len(nodes_values) + 1

    # Determine grid dimensions based on total plots.
    n_cols = math.ceil(math.sqrt(total_plots))
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=False)
    axes = axes.flatten()

    # Plot the line charts for each Nodes value.
    for ax, nodes in zip(axes, nodes_values):
        sub_df = grouped[grouped["Nodes"] == nodes]
        for idx, coeff in enumerate(coefficients):
            coeff_df = sub_df[sub_df["Coefficient"] == coeff].sort_values(by="Epsilon")
            ax.plot(coeff_df["Epsilon"], coeff_df[y_value], marker='o', linestyle='-', color=random_colors[idx % len(random_colors)])
        ax.set_title(f"Nodes = {nodes}")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel(y_label_title)

    # Create dummy handles for the legend using our random colors.
    legend_handles = []
    for idx, coeff in enumerate(coefficients):
        color = random_colors[idx % len(random_colors)]
        line = plt.Line2D([0], [0], marker='o', color=color, linestyle='-', label=f"Density: {coeff / 2} Ln(N)")
        legend_handles.append(line)

    # The extra subplot (last cell) is used for the legend.
    legend_ax = axes[len(nodes_values)]
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_handles, loc='center', frameon=False)

    # Remove any unused subplots if they exist.
    for ax in axes[total_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)

# --------------------- Create Figures using all coefficients ---------------------

# Figure for Oracle_MIS using all coefficients.
create_line_chart_figure(df, y_value="Oracle_MIS", filename="eval_line_chart_with_legend_Oracle.png", y_label_title="Oracle_MIS")

# Figure for Greedy_MIS using all coefficients.
create_line_chart_figure(df, y_value="Greedy_MIS", filename="eval_line_chart_with_legend_Greedy.png", y_label_title="Greedy_MIS")

# --------------------- Create Figures for a subset of coefficients ---------------------
subset_coeffs = [2, 5, 10, 15, 20, 30]

# Figure for Oracle_MIS using only coefficients 1, 5, 10, and 15.
create_line_chart_figure(df, y_value="Oracle_MIS", filename="eval_line_chart_with_legend_Oracle_subset.png", y_label_title="Oracle_MIS", coefficients_subset=subset_coeffs)

# Figure for Greedy_MIS using only coefficients 1, 5, 10, and 15.
create_line_chart_figure(df, y_value="Greedy_MIS", filename="eval_line_chart_with_legend_Greedy_subset.png", y_label_title="Greedy_MIS", coefficients_subset=subset_coeffs)
