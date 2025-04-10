import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap

# Find all CSV files with the specified prefix.
csv_files = glob.glob("bipartite_experiment_results_epislons_*.csv")

# Load and concatenate all CSV files.
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Create a new column for normalized improvement.
df["ApproximationScore"] = df["Oracle_MIS"] / df["GroundTruth_MIS"] * 100

# Group by Epsilon, Nodes, and Coefficient and compute the average NormalizedImprovement.
grouped = df.groupby(["Epsilon", "Nodes", "Coefficient"], as_index=False)["ApproximationScore"].mean()

# Determine the global min and max normalized improvement for consistent color scale.
global_min = grouped["ApproximationScore"].min()
global_max = grouped["ApproximationScore"].max()

# Get unique epsilon values.
epsilons = sorted(df["Epsilon"].unique())

# Determine subplot grid: 2 rows and as many columns as needed.
n_eps = len(epsilons)
n_cols = math.ceil(n_eps / 2)

fig, axes = plt.subplots(2, n_cols, figsize=(10 * n_cols, 10), sharey=True)
axes = axes.flatten()  # Flatten to iterate easily

# For each epsilon, pivot the table and plot the heatmap with the same color scale.
for ax, eps in zip(axes, epsilons):
    sub_df = grouped[grouped["Epsilon"] == eps]
    pivot = sub_df.pivot(index="Coefficient", columns="Nodes", values="ApproximationScore")
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="viridis", 
                vmin=global_min, vmax=global_max)
    ax.set_title(f"Percent Improvement with Oracle (At Epsilon = {eps})")
    ax.set_xlabel("Nodes in Evenly Sized Bipartite Graph")
    ax.set_ylabel("X * Log(n) / N")

# If there are any unused subplots, remove them.
for ax in axes[len(epsilons):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.savefig("evalApprox.png")
plt.show()

X = grouped[["Nodes", "Coefficient", "Epsilon"]]
y = grouped["ApproximationScore"]

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the random forest regressor.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model.
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the predicted vs actual values.
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, 
         label=f'Ideal Fit (R² = {r2:.3f})')
plt.xlabel("Actual Normalized Improvement")
plt.ylabel("Predicted Normalized Improvement")
plt.title("Predicted vs. Actual Normalized Improvement")
plt.legend()
plt.savefig('evalApproxFit.png')
plt.show()


# 1. Feature Importances
importances = rf_model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# 2. SHAP Analysis
# Create an explainer and compute SHAP values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Generate a summary plot of SHAP values
shap.summary_plot(shap_values, X_test, feature_names=X.columns)