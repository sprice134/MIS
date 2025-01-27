#!/usr/bin/env python3

import os
import json
import networkx as nx
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

def load_graph(file_path, n=5):
    """
    Load a graph from an edge list file and ensure it has exactly `n` nodes,
    labeled from 0 to n-1. If any node is missing, we add it (isolated).
    """
    G = nx.read_edgelist(file_path, nodetype=int)
    # Ensure all nodes [0 .. n-1] exist
    G.add_nodes_from(range(n))
    return G

def all_maximum_independent_sets_bruteforce(G):
    """
    Return a list of ALL maximum independent sets for graph G.
    Each set is returned as a Python set of node indices.
    
    This is a purely brute-force, exponential-time approach, suitable for small graphs.
    """

    # Base case: If G is empty, the only "independent set" is the empty set.
    if len(G) == 0:
        return [set()]

    # Pick one node
    v = next(iter(G.nodes()))

    # 1) Exclude v
    G_exclude = G.copy()
    G_exclude.remove_node(v)
    exclude_sets = all_maximum_independent_sets_bruteforce(G_exclude)

    # 2) Include v (which means we must remove its neighbors as well)
    G_include = G.copy()
    neighbors = list(G_include.neighbors(v))
    G_include.remove_node(v)
    G_include.remove_nodes_from(neighbors)
    include_sets = all_maximum_independent_sets_bruteforce(G_include)
    # Add v back to each of those sets
    include_sets = [s.union({v}) for s in include_sets]

    # Combine all sets
    combined = exclude_sets + include_sets

    # However, 'combined' will contain *all* independent sets of various sizes.
    # We only want the ones that have the maximum possible size.
    max_size = max(len(s) for s in combined)
    # Filter down to those that match `max_size`
    all_mis = [s for s in combined if len(s) == max_size]
    return all_mis

def process_graph(file_path, n=5):
    """
    Worker function to process a single graph file.
    It returns a dictionary with:
      - file_path: the input file path
      - MIS_SIZE: (int) size of the maximum independent set
      - MIS_SETS: (list of lists) all possible MIS sets
      - MIS_CELLS: (list of 0/1) 1 if that node appears in *any* MIS, 0 otherwise
      - MIS_CELLS_PROB: (list of floats) fraction of MIS sets that include that node
    """
    try:
        G = load_graph(file_path, n=n)
        all_mis_sets = all_maximum_independent_sets_bruteforce(G)

        if len(all_mis_sets) == 0:
            # Graph has no nodes, or something unexpected
            # In practice, for n>0, this shouldn't happen.
            return {
                "file_path": file_path,
                "MIS_SIZE": 0,
                "MIS_SETS": [],
                "MIS_CELLS": [0]*n,
                "MIS_CELLS_PROB": [0.0]*n
            }

        # All sets in all_mis_sets have the same size, by definition
        mis_size = len(next(iter(all_mis_sets)))

        # Convert each set to a sorted list for JSON-serialization
        mis_sets_list = [sorted(list(s)) for s in all_mis_sets]

        # Build MIS_CELLS: 1 if node appears in ANY MIS, else 0
        appears_any = set().union(*all_mis_sets)  # union of all sets
        mis_cells = [1 if i in appears_any else 0 for i in range(n)]

        # Build MIS_CELLS_PROB: fraction of MIS sets that include each node
        num_mis_sets = len(all_mis_sets)
        # Count how many MIS sets contain node i
        contain_counts = [0]*n
        for s in all_mis_sets:
            for node in s:
                contain_counts[node] += 1
        
        # Probability = contain_counts[i] / num_mis_sets
        mis_cells_prob = [
            contain_counts[i]/num_mis_sets for i in range(n)
        ]

        return {
            "file_path": file_path,
            "MIS_SIZE": mis_size,
            "MIS_SETS": mis_sets_list,
            "MIS_CELLS": mis_cells,
            "MIS_CELLS_PROB": mis_cells_prob
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "MIS_SIZE": None,
            "MIS_SETS": [],
            "MIS_CELLS": [],
            "MIS_CELLS_PROB": []
        }

def main():
    parser = argparse.ArgumentParser(
        description="Compute all Maximum Independent Sets (MIS) for small .edgelist graphs."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="Number of nodes in the graphs (default: 5)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing all .edgelist files. "
             "If not specified, defaults to 'noniso_{n}_networkx'"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Name of the JSON file to store the results. "
             "If not specified, defaults to 'mis_results_{n}.json'"
    )
    args = parser.parse_args()

    n = args.nodes
    input_dir = args.input_dir if args.input_dir else f"noniso_{n}_networkx"
    output_file = args.output_file if args.output_file else f"mis_results_{n}.json"

    # Verify the directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return

    # Gather all .edgelist files from the directory
    edgelist_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".edgelist")
    ]

    if not edgelist_files:
        print(f"No .edgelist files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(edgelist_files)} .edgelist files to process in '{input_dir}'.")
    
    # Use all available CPU cores (or modify if you prefer fewer)
    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes.")

    results = []
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to process files in parallel
        for result in pool.imap_unordered(partial(process_graph, n=n), edgelist_files):
            results.append(result)

    print(f"All tasks processed. Total results collected: {len(results)}")

    # Save all results in a single JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
    # python compute_all_mis_sets.py --nodes 3 --input_dir ../NonIsoEval/noniso_3_networkx --output_file all_mis_results_3.json
    # python compute_all_mis_sets.py --nodes 4 --input_dir ../NonIsoEval/noniso_4_networkx --output_file all_mis_results_4.json
    # python compute_all_mis_sets.py --nodes 5 --input_dir ../NonIsoEval/noniso_5_networkx --output_file all_mis_results_5.json
    # python compute_all_mis_sets.py --nodes 6 --input_dir ../NonIsoEval/noniso_6_networkx --output_file all_mis_results_6.json
    # python compute_all_mis_sets.py --nodes 7 --input_dir ../NonIsoEval/noniso_7_networkx --output_file all_mis_results_7.json
