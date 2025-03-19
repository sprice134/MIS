#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

def load_and_process_csv(filename):
    # Load CSV file into DataFrame
    df = pd.read_csv(filename)
    # Compute absolute error and squared error
    df['abs_error'] = (df['bravemen_mis'] - df['true_mis']).abs()
    df['sq_error'] = (df['bravemen_mis'] - df['true_mis'])**2
    return df

def compute_rmse(df):
    # RMSE = sqrt(mean(square_error))
    rmse = np.sqrt(df['sq_error'].mean())
    return rmse

def compare_errors(df1, df2):
    # Merge on 'graph_id' (or another unique identifier)
    merged = pd.merge(df1, df2, on='graph_id', suffixes=('_0.1', '_0.3'))
    
    # For each graph, compare absolute errors from both csvs.
    ident = (merged['abs_error_0.1'] == merged['abs_error_0.3']).sum()
    better_0_1 = (merged['abs_error_0.1'] < merged['abs_error_0.3']).sum()
    better_0_3 = (merged['abs_error_0.1'] > merged['abs_error_0.3']).sum()
    
    return ident, better_0_1, better_0_3, merged.shape[0]

def main():
    parser = argparse.ArgumentParser(
        description="Compare MIS CSV results for epsilon=0.1 and epsilon=0.3"
    )
    parser.add_argument("--csv_0_1", type=str, required=True,
                        help="Path to CSV file with epsilon=0.1 results")
    parser.add_argument("--csv_0_3", type=str, required=True,
                        help="Path to CSV file with epsilon=0.3 results")
    
    args = parser.parse_args()
    
    # Load CSV files
    df_0_1 = load_and_process_csv(args.csv_0_1)
    df_0_3 = load_and_process_csv(args.csv_0_3)
    
    # Compute RMSE for each
    rmse_0_1 = compute_rmse(df_0_1)
    rmse_0_3 = compute_rmse(df_0_3)
    
    print(f"RMSE for epsilon=0.1: {rmse_0_1:.4f}")
    print(f"RMSE for epsilon=0.3: {rmse_0_3:.4f}")
    
    # Compare per-graph absolute errors between the two CSVs
    ident, better_0_1, better_0_3, total = compare_errors(df_0_1, df_0_3)
    
    print(f"\nOut of {total} graphs:")
    print(f"  - {ident} graphs had identical absolute error.")
    print(f"  - {better_0_1} graphs had lower error for epsilon=0.1.")
    print(f"  - {better_0_3} graphs had lower error for epsilon=0.3.")

if __name__ == "__main__":
    main()

    '''
    python eval.py --csv_0_1 v3_mis_basic_results.csv --csv_0_3 v2_mis_basic_results_0.2.csv

    '''
