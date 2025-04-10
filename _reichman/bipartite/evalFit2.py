import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import shap

# Load CSV files and concatenate them.
csv_files = glob.glob("bipartite_experiment_results_epislons_*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Create a new column that represents the percentage of the true MIS that is found.
df["TrueMIS_Percentage"] = df["Oracle_MIS"] / df["GroundTruth_MIS"] * 100

# Group by Epsilon, Nodes, and Coefficient and compute the average TrueMIS_Percentage.
grouped = df.groupby(["Epsilon", "Nodes", "Coefficient"], as_index=False)["TrueMIS_Percentage"].mean()

# Define features and target.
X = grouped[["Nodes", "Coefficient", "Epsilon"]]
y = grouped["TrueMIS_Percentage"]

# Create a group identifier as a string for each row based on "Nodes", "Coefficient", and "Epsilon".
groups = X["Nodes"].astype(str) + "_" + X["Coefficient"].astype(str) + "_" + X["Epsilon"].astype(str)

# Use GroupShuffleSplit so that no group appears in both train and test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

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
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label=f'Ideal Fit (R² = {r2:.3f})')
plt.xlabel("Actual TrueMIS Percentage")
plt.ylabel("Predicted TrueMIS Percentage")
plt.title("Predicted vs. Actual TrueMIS Percentage")
plt.legend()
plt.savefig('evalFit_TrueMIS.png')
plt.show()

# 1. Feature Importances
importances = rf_model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# 2. SHAP Analysis
# Create an explainer and compute SHAP values.
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Generate a summary plot of SHAP values.
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
