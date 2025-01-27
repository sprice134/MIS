#!/usr/bin/env python3

# python analysis.py --nodes 7 --gt mis_results_7.json --greedy greedy_mis_results_7.json

import json
import argparse
import os

def load_results(json_file):
    """
    Loads the JSON file and returns a dictionary keyed by file_path.
    Each value is another dictionary with the relevant MIS data.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Build a dictionary keyed by file_path
    results_dict = {}
    for entry in data:
        file_path = entry.get("file_path")
        if file_path:
            results_dict[file_path] = entry
    return results_dict

def main():
    parser = argparse.ArgumentParser(
        description="Compare brute-force MIS results to greedy MIS results."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        required=True,
        help="Number of nodes in the graphs."
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Path to the JSON file produced by the brute-force MIS script (e.g., mis_results_{n}.json). If not specified, defaults to 'mis_results_{n}.json'"
    )
    parser.add_argument(
        "--greedy",
        type=str,
        default=None,
        help="Path to the JSON file produced by the greedy MIS script (e.g., greedy_mis_results_{n}.json). If not specified, defaults to 'greedy_mis_results_{n}.json'"
    )
    args = parser.parse_args()

    n = args.nodes
    brute_force_file = args.gt if args.gt else f"mis_results_{n}.json"
    greedy_file = args.greedy if args.greedy else f"greedy_mis_results_{n}.json"

    # Validate the files exist
    if not os.path.isfile(brute_force_file):
        print(f"Error: file '{brute_force_file}' does not exist.")
        return
    if not os.path.isfile(greedy_file):
        print(f"Error: file '{greedy_file}' does not exist.")
        return

    # Load both JSON files into dictionaries keyed by file_path
    brute_force_results = load_results(brute_force_file)
    greedy_results = load_results(greedy_file)

    # We'll track how many mismatch vs. total
    total_graphs_compared = 0
    mismatch_count = 0

    # Iterate over all file paths in the brute_force_results
    for file_path, bf_entry in brute_force_results.items():
        if file_path not in greedy_results:
            # If it's missing in the greedy results, we can't compare
            continue
        
        gf_entry = greedy_results[file_path]
        total_graphs_compared += 1

        # Extract sets for easier comparison
        # Key name in brute_force is "mis_size"
        bf_mis_size = bf_entry.get("mis_size")

        # Key name in greedy results is "greedy_mis_size"
        gf_mis_size = gf_entry.get("greedy_mis_size")

        # Compare
        if bf_mis_size != gf_mis_size:
            mismatch_count += 1
            print(f"DIFFERENT MIS for: {file_path}")

    print(f"\nTotal graphs compared: {total_graphs_compared}")
    print(f"Number of mismatches: {mismatch_count}")

if __name__ == "__main__":
    main()
