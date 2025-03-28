import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find all CSV files with the specified prefix.
csv_files = glob.glob("bipartite_experiment_results_epislons_*.csv")

# Load and concatenate all CSV files.
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Create a new column for normalized improvement.
df["NormalizedImprovement"] = df["Improvement"] / df["Greedy_MIS"] * 100

# Group by Epsilon, Nodes, and Coefficient and compute the average NormalizedImprovement.
grouped = df.groupby(["Epsilon", "Nodes", "Coefficient"], as_index=False)["NormalizedImprovement"].mean()

# Determine the global min and max normalized improvement for consistent color scale.
global_min = grouped["NormalizedImprovement"].min()
global_max = grouped["NormalizedImprovement"].max()

# Get unique epsilon values.
epsilons = sorted(df["Epsilon"].unique())

# Determine subplot grid: 2 rows and as many columns as needed.
n_eps = len(epsilons)
n_cols = math.ceil(n_eps / 2)

fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 10), sharey=True)
axes = axes.flatten()  # Flatten to iterate easily

# For each epsilon, pivot the table and plot the heatmap with the same color scale.
for ax, eps in zip(axes, epsilons):
    sub_df = grouped[grouped["Epsilon"] == eps]
    pivot = sub_df.pivot(index="Coefficient", columns="Nodes", values="NormalizedImprovement")
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="viridis", 
                vmin=global_min, vmax=global_max)
    ax.set_title(f"Percent Improvement with Oracle (At Epsilon = {eps})")
    ax.set_xlabel("Nodes in Evenly Sized Bipartite Graph")
    ax.set_ylabel("X * Log(n) / N")

# If there are any unused subplots, remove them.
for ax in axes[len(epsilons):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.savefig("eval.png")
plt.show()
