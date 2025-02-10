#!/usr/bin/env python3
"""
Optimized MIS evaluator

This script processes graph files (in edge list format) by:
  - Loading the graph (ensuring isolated nodes are added).
  - Computing a lower bound for the maximum independent set (MIS) using a
    fast minimum-degree greedy algorithm.
  - Iteratively searching for independent sets of increasing size until none exist,
    thereby identifying all maximum independent sets.
  - Computing per-node statistics:
      * MIS_CELLS: 1 if the node appears in any maximum independent set; else 0.
      * MIS_CELLS_PROB: Fraction of all maximum independent sets that include each node.
  - Grouping and saving the results into JSON files as soon as all tasks for a given
    node count and removal percentage are complete.
  
Usage examples are provided at the end.
"""

import os
import json
import networkx as nx
import random
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import argparse

# -------------------------
# Graph Loading and Greedy MIS
# -------------------------

def load_graph(file_path):
    """
    Load a graph from an edge list file, ensuring that lines with a single token
    become isolated nodes and lines with two tokens become edges.
    """
    edges = []
    isolated_nodes = set()
    max_node_id = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            parts = line.split()
            if len(parts) == 2:
                u, v = map(int, parts)
                edges.append((u, v))
                max_node_id = max(max_node_id, u, v)
            elif len(parts) == 1:
                node_id = int(parts[0])
                isolated_nodes.add(node_id)
                max_node_id = max(max_node_id, node_id)
            else:
                print(f"Warning: ignoring malformed line in {file_path}: {line}")

    G = nx.Graph()
    # Add all nodes up to max_node_id
    G.add_nodes_from(range(max_node_id + 1))
    G.add_edges_from(edges)
    # Ensure isolated nodes are added (redundant, but safe)
    for node in isolated_nodes:
        G.add_node(node)
    return G

def greedy_independent_set(G):
    """
    Compute an independent set using a minimum-degree greedy algorithm.
    Returns a set of nodes.
    """
    H = G.copy()
    indep_set = set()
    while H.number_of_nodes() > 0:
        # Choose the node with minimum degree
        node, _ = min(H.degree, key=lambda x: x[1])
        indep_set.add(node)
        # Remove node and its neighbors from H
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return indep_set

# -------------------------
# Iterative Deepening for MIS
# -------------------------

def find_independent_sets_of_size(G, k):
    """
    Recursively find all independent sets of size exactly k in graph G.
    Uses backtracking with a simple bound:
      - If current set size + remaining candidates < k, prune.
    
    Returns a list of sets (each is a set of nodes).
    """
    result = []
    nodes = list(G.nodes())

    def backtrack(current, candidates, start):
        # Prune if even with all remaining candidates we cannot reach k.
        if len(current) + len(candidates) - start < k:
            return
        if len(current) == k:
            result.append(set(current))
            return
        for i in range(start, len(candidates)):
            node = candidates[i]
            # Check if node is independent with respect to current
            if any(neighbor in current for neighbor in G.neighbors(node)):
                continue
            backtrack(current + [node], candidates, i + 1)
    
    backtrack([], nodes, 0)
    return result

def find_maximum_independent_sets(G):
    """
    First use a greedy algorithm to get a lower bound for the size of a MIS.
    Then, iteratively search for independent sets of increasing size until
    none are found. Returns the list of all maximum independent sets.
    """
    # Get a greedy solution as a lower bound.
    greedy_set = greedy_independent_set(G)
    lower_bound = len(greedy_set)
    max_sets = []
    k = lower_bound  # start search at the greedy size

    while True:
        # Find all independent sets of size k.
        sets_k = find_independent_sets_of_size(G, k)
        if sets_k:
            max_sets = sets_k  # update maximum sets
            k += 1
        else:
            # No independent set of size k exists, so maximum size is k-1.
            break
    return max_sets

# -------------------------
# Process Single Graph (Worker)
# -------------------------

def process_graph(args):
    """Worker function to process a single graph file."""
    file_path, n, percent, iteration = args
    try:
        G = load_graph(file_path)
        max_ind_sets = find_maximum_independent_sets(G)
        if not max_ind_sets:
            mis_size = 0
            mis_sets = []
            mis_cells = [0] * n
            mis_cells_prob = [0.0] * n
        else:
            mis_size = len(next(iter(max_ind_sets)))  # All max sets have same size.
            mis_sets = [sorted(list(s)) for s in max_ind_sets]

            # MIS_CELLS: 1 if node appears in any maximum independent set; else 0.
            union_sets = set().union(*max_ind_sets)
            mis_cells = [1 if i in union_sets else 0 for i in range(n)]

            # MIS_CELLS_PROB: Fraction of maximum independent sets that include each node.
            num_sets = len(max_ind_sets)
            counts = [0] * n
            for s in max_ind_sets:
                for node in s:
                    counts[node] += 1
            mis_cells_prob = [counts[i] / num_sets for i in range(n)]
        
        result = {
            "file_path": file_path,
            "num_nodes": n,
            "percent_removed": percent,
            "iteration": iteration,
            "MIS_SIZE": mis_size,
            "MIS_SETS": mis_sets,
            "MIS_CELLS": mis_cells,
            "MIS_CELLS_PROB": mis_cells_prob
        }
        print(f"Processed {file_path}: MIS size = {mis_size}, # of MIS sets = {len(mis_sets)}")
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "num_nodes": n,
            "percent_removed": percent,
            "iteration": iteration,
            "MIS_SIZE": None,
            "MIS_SETS": [],
            "MIS_CELLS": [],
            "MIS_CELLS_PROB": []
        }

# -------------------------
# Task Gathering and Result Saving
# -------------------------

def gather_tasks(base_dir, node_counts_list, removal_percents, iterations):
    """Traverse directory structure to gather tasks based on specified node counts and removal percents."""
    tasks = []
    for node_folder in os.listdir(base_dir):
        node_folder_path = os.path.join(base_dir, node_folder)
        if not os.path.isdir(node_folder_path):
            continue

        # Extract node count from folder name (assume "nodes_{n}")
        try:
            n = int(node_folder.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Unable to extract node count from folder '{node_folder}'. Skipping.")
            continue

        if n not in node_counts_list:
            continue

        for percent_folder in os.listdir(node_folder_path):
            percent_folder_path = os.path.join(node_folder_path, percent_folder)
            if not os.path.isdir(percent_folder_path):
                continue

            try:
                percent_str = percent_folder.split('_')[1]
                percent = int(percent_str.replace('percent', ''))
            except (IndexError, ValueError):
                print(f"Warning: Unable to extract removal percent from folder '{percent_folder}'. Skipping.")
                continue

            if percent not in removal_percents:
                continue

            for graph_file in os.listdir(percent_folder_path):
                if not graph_file.endswith(".edgelist"):
                    continue
                file_path = os.path.join(percent_folder_path, graph_file)
                try:
                    iteration_str = graph_file.split('_')[1]
                    iteration = int(iteration_str.split('.')[0])
                except (IndexError, ValueError):
                    print(f"Warning: Unable to extract iteration from file '{graph_file}'. Assigning iteration as None.")
                    iteration = None

                tasks.append((file_path, n, percent, iteration))
    return tasks

def save_group_result(n, percent, results, output_dir):
    """
    Save results for a given node count and removal percentage group into a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_filename = f"nodes_{n}_removal_{percent}percent.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for Nodes={n}, Removal={percent}% to '{json_path}'")
    except Exception as e:
        print(f"Error saving JSON file '{json_path}': {e}")

# -------------------------
# Main function
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute maximum independent sets (MIS) for .edgelist graphs using an optimized iterative deepening approach."
    )
    parser.add_argument("--node_counts", type=int, nargs='+', required=True,
                        help="List of node counts to process, e.g., --node_counts 15 20 25")
    parser.add_argument("--removal_percents", type=int, nargs='+', default=list(range(15, 90, 5)),
                        help="List of removal percentages to process, e.g., --removal_percents 15 20 25")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations per combination of node count and removal percent")
    parser.add_argument("--base_dir", type=str, default="generated_graphs",
                        help="Base directory where generated graphs are stored")
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped",
                        help="Directory to save the grouped JSON result files")
    args = parser.parse_args()

    print(f"Processing node counts: {args.node_counts}")
    print(f"Processing removal percentages: {args.removal_percents}")
    print(f"Iterations per combination: {args.iterations}")
    print(f"Base directory: '{args.base_dir}', Output directory: '{args.output_dir}'")
    
    tasks = gather_tasks(args.base_dir, args.node_counts, args.removal_percents, args.iterations)
    print(f"Total tasks to process: {len(tasks)}")
    if not tasks:
        print("No tasks found. Check your directory structure and parameters.")
        return

    # Determine the expected number of results for each (n, percent) group.
    group_expected = defaultdict(int)
    for task in tasks:
        _, n, percent, _ = task
        group_expected[(n, percent)] += 1

    # Dictionary to accumulate results by group.
    group_results = defaultdict(list)

    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes.")
    
    total_results = 0
    with Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(process_graph, tasks):
            total_results += 1
            key = (res["num_nodes"], res["percent_removed"])
            group_results[key].append(res)
            # When a group is complete, save its JSON file.
            if len(group_results[key]) == group_expected[key]:
                save_group_result(key[0], key[1], group_results[key], args.output_dir)
                # Optionally remove the group from the dictionary to free memory.
                del group_results[key]

    print(f"All tasks processed. Total results collected: {total_results}")
    
    # Save any remaining groups (if any tasks failed to be grouped, for example)
    if group_results:
        print("Saving any remaining groups that were not already saved...")
        for (n, percent), res_list in group_results.items():
            save_group_result(n, percent, res_list, args.output_dir)
    
    print("All grouped results have been saved.")

if __name__ == "__main__":
    main()

"""
Example usage:
    python misEvaluator_optimized.py \
        --node_counts 60 80 \
        --iterations 50 \
        --base_dir test_generated_graphs \
        --output_dir test_mis_results_grouped_v3

Other example commands (adapt as needed):
    python misEvaluator_optimized_working.py --node_counts 95 --base_dir test_generated_graphs --output_dir test_mis_results_grouped_v3
"""
