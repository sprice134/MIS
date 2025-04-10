import itertools
import numpy as np
import pandas as pd

# Define the parameter ranges.
# Coefficients: even numbers from 2 to 28 plus additional values 50, 72, and 100.
coefficients = list(range(2, 30, 2)) + [50, 72, 100]

# Nodes: from 10000 to 190000 (step 10000) plus additional values [2500, 5000]
nodes = list(range(10000, 200000, 10000)) + [2500, 5000]

# Epsilon: from 0 to 0.25 in steps of 0.025.
# We add a tiny tolerance so that 0.25 is included if desired.
epsilons = list(np.arange(0, 0.25 + 1e-8, 0.0125))

# Number of trials: two (trial IDs 0 and 1)
trials = [0, 1]

# Use itertools.product to create all parameter combinations.
combinations = list(itertools.product(coefficients, nodes, trials, epsilons))

# Create a list of dictionaries (one per combination). For each row the Seed is 42 + trial.
doe_rows = []
for coeff, node_val, trial, epsilon in combinations:
    row = {
        "Coefficient": coeff,
        "Nodes": node_val,
        "Trial": trial,
        "Epsilon": epsilon,
        "Seed": 42 + trial
    }
    doe_rows.append(row)

# Create a DataFrame.
doe_df = pd.DataFrame(doe_rows)

# (Optional) If you want to also include empty columns for the results, you could add them.
# For example:
for col in ["GroundTruth_MIS", "Greedy_MIS", "Size_S", "Oracle_MIS", "Improvement"]:
    doe_df[col] = None

# Save the DOE file.
doe_csv_filename = "experimentalDOE.csv"
doe_df.to_csv(doe_csv_filename, index=False)
print("DOE file saved to", doe_csv_filename)
