import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap

# Load CSV files and concatenate them.
csv_files = glob.glob("bipartite_experiment_results_epislons_*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Create normalized improvement column.
df["NormalizedImprovement"] = df["Improvement"] / df["Greedy_MIS"] * 100

# Group by Epsilon, Nodes, and Coefficient and compute the average NormalizedImprovement.
grouped = df.groupby(["Epsilon", "Nodes", "Coefficient"], as_index=False)["NormalizedImprovement"].mean()

# Define features and target.
X = grouped[["Nodes", "Coefficient", "Epsilon"]]
y = grouped["NormalizedImprovement"]

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
plt.show()
plt.savefig('evalFit.png')

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



