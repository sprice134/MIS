#!/usr/bin/env python3

import os
import json
import networkx as nx
import random
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from functools import partial
import argparse

def load_graph(file_path):
    """
    Load a graph from an edge list file, ensuring single-token lines
    become isolated nodes, and multi-token lines become edges.
    """
    edges = []
    isolated_nodes = set()
    max_node_id = 0

    # 1) Parse the file line by line ourselves
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            parts = line.split()
            if len(parts) == 2:
                # Format: "u v"
                u, v = map(int, parts)
                edges.append((u, v))
                max_node_id = max(max_node_id, u, v)
            elif len(parts) == 1:
                # Single token -> an isolated node
                node_id = int(parts[0])
                isolated_nodes.add(node_id)
                max_node_id = max(max_node_id, node_id)
            else:
                # Skip or warn about unexpected line formats
                print(f"Warning: ignoring malformed line in {file_path}: {line}")

    # 2) Build the graph
    G = nx.Graph()
    # Add all nodes from 0..max_node_id so that "missing" nodes also appear
    G.add_nodes_from(range(max_node_id + 1))
    # Add edges
    for (u, v) in edges:
        G.add_edge(u, v)
    # Also ensure single-token nodes are definitely included
    for node in isolated_nodes:
        G.add_node(node)

    return G


def all_maximum_independent_sets_bruteforce(G):
    """
    Return a list of ALL maximum independent sets for graph G.
    Each set is returned as a Python set of node indices.
    
    This is a purely brute-force, exponential-time approach, suitable for small graphs.
    """
    # Base case: If G is empty, the only "independent set" is the empty set.
    if len(G.nodes()) == 0:
        return [set()]
    
    # Pick one node
    v = next(iter(G.nodes()))
    
    # Branch 1: Exclude v
    G_exclude = G.copy()
    G_exclude.remove_node(v)
    mis_exclude = all_maximum_independent_sets_bruteforce(G_exclude)
    
    # Branch 2: Include v (and remove its neighbors)
    G_include = G.copy()
    neighbors = list(G_include.neighbors(v))
    G_include.remove_node(v)
    G_include.remove_nodes_from(neighbors)
    mis_include = all_maximum_independent_sets_bruteforce(G_include)
    # Add v back to each of those sets
    mis_include = [s.union({v}) for s in mis_include]
    
    # Combine all sets
    combined = mis_exclude + mis_include
    
    # Find the size of the maximum independent sets
    max_size = max(len(s) for s in combined) if combined else 0
    
    # Filter to include only the maximum independent sets
    all_mis = [s for s in combined if len(s) == max_size]
    return all_mis

def process_graph(args):
    """Worker function to process a single graph file."""
    file_path, n, percent, iteration = args
    try:
        G = load_graph(file_path)
        all_mis_sets = all_maximum_independent_sets_bruteforce(G)
        if not all_mis_sets:
            mis_size = 0
            mis_sets = []
            mis_cells = [0]*n
            mis_cells_prob = [0.0]*n
        else:
            mis_size = len(all_mis_sets[0])  # All MIS sets have the same size
            # Convert each set to a sorted list for JSON serialization
            mis_sets = [sorted(list(s)) for s in all_mis_sets]
            
            # MIS_CELLS: 1 if node appears in ANY MIS, else 0
            appears_any = set().union(*all_mis_sets)
            mis_cells = [1 if i in appears_any else 0 for i in range(n)]
            
            # MIS_CELLS_PROB: Fraction of MIS sets that include each node
            num_mis_sets = len(all_mis_sets)
            contain_counts = [0]*n
            for s in all_mis_sets:
                for node in s:
                    contain_counts[node] += 1
            mis_cells_prob = [
                contain_counts[i]/num_mis_sets if num_mis_sets > 0 else 0.0
                for i in range(n)
            ]
        
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
        print(f"Processed {file_path}: MIS size = {mis_size}, Number of MIS sets = {len(mis_sets)}")
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

def gather_tasks(base_dir, node_counts_list, removal_percents, iterations):
    """Traverse directory structure to gather tasks based on specified node counts and removal percents."""
    tasks = []
    for node_folder in os.listdir(base_dir):
        node_folder_path = os.path.join(base_dir, node_folder)
        if not os.path.isdir(node_folder_path):
            continue

        # Extract node count from folder name (assuming format "nodes_{n}")
        try:
            n = int(node_folder.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Unable to extract node count from folder '{node_folder}'. Skipping.")
            continue

        if n not in node_counts_list:
            continue  # Skip node counts not in the specified list

        for percent_folder in os.listdir(node_folder_path):
            percent_folder_path = os.path.join(node_folder_path, percent_folder)
            if not os.path.isdir(percent_folder_path):
                continue

            # Extract percent removed from folder name (assuming format "removal_{percent}percent")
            try:
                percent_str = percent_folder.split('_')[1]
                percent = int(percent_str.replace('percent', ''))
            except (IndexError, ValueError):
                print(f"Warning: Unable to extract removal percent from folder '{percent_folder}'. Skipping.")
                continue

            if percent not in removal_percents:
                continue  # Skip removal percents not in the specified list

            for graph_file in os.listdir(percent_folder_path):
                if not graph_file.endswith(".edgelist"):
                    continue

                file_path = os.path.join(percent_folder_path, graph_file)

                # Extract iteration from filename (assuming format "graph_{iteration}.edgelist")
                try:
                    iteration_str = graph_file.split('_')[1]
                    iteration = int(iteration_str.split('.')[0])
                except (IndexError, ValueError):
                    print(f"Warning: Unable to extract iteration from file '{graph_file}'. Assigning iteration as None.")
                    iteration = None

                tasks.append((file_path, n, percent, iteration))
    return tasks

def save_grouped_results(results, output_dir):
    """
    Group results by node count and percent removed,
    then save each group into separate JSON files.
    """
    grouped = defaultdict(lambda: defaultdict(list))  # {n: {percent: [results]}}
    
    for result in results:
        n = result["num_nodes"]
        percent = result["percent_removed"]
        grouped[n][percent].append(result)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for n, percent_dict in grouped.items():
        for percent, res_list in percent_dict.items():
            # Define the JSON filename
            json_filename = f"nodes_{n}_removal_{percent}percent.json"
            json_path = os.path.join(output_dir, json_filename)
            
            # Save the list of results to the JSON file
            try:
                with open(json_path, "w") as f:
                    json.dump(res_list, f, indent=2)
                print(f"Saved results for Nodes={n}, Removal={percent}% to '{json_path}'")
            except Exception as e:
                print(f"Error saving JSON file '{json_path}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute all Maximum Independent Sets (MIS) for generated .edgelist graphs."
    )
    parser.add_argument(
        "--node_counts",
        type=int,
        nargs='+',
        required=True,
        help="List of node counts to process, e.g., --node_counts 15 20 25"
    )
    parser.add_argument(
        "--removal_percents",
        type=int,
        nargs='+',
        default=list(range(15, 90, 5)),
        help="List of edge removal percentages to process, e.g., --removal_percents 15 20 25"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per combination of node count and removal percent"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="generated_graphs",
        help="Base directory where generated graphs are stored"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mis_results_grouped",
        help="Directory to save the grouped JSON result files"
    )
    args = parser.parse_args()

    node_counts_list = args.node_counts
    removal_percents = args.removal_percents
    iterations = args.iterations
    base_dir = args.base_dir
    output_dir = args.output_dir

    # Validate input node counts and removal percents
    if not node_counts_list:
        print("Error: At least one node count must be specified.")
        return
    if not removal_percents:
        print("Error: At least one removal percent must be specified.")
        return

    print(f"Node counts to process: {node_counts_list}")
    print(f"Removal percentages to process: {removal_percents}%")
    print(f"Number of iterations per combination: {iterations}")
    print(f"Base directory: '{base_dir}'")
    print(f"Output directory: '{output_dir}'")

    # Gather tasks based on specified node counts and removal percents
    tasks = gather_tasks(base_dir, node_counts_list, removal_percents, iterations)
    print(f"Total tasks to process: {len(tasks)}")

    if not tasks:
        print("No tasks found. Please check your directory structure and input parameters.")
        return

    # Determine the number of processes to use
    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes.")

    results = []
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to process files in parallel
        for result in pool.imap_unordered(process_graph, tasks):
            results.append(result)

    print(f"All tasks processed. Total results collected: {len(results)}")

    # Save grouped results into separate JSON files
    save_grouped_results(results, output_dir)

    print("All grouped results have been saved.")

if __name__ == "__main__":
    main()
#     python misEvaluator.py \
#   --node_counts 15 20 25 \
#   --removal_percents 15 20 25 \
#   --iterations 50 \
#   --base_dir generated_graphs \
#   --output_dir mis_results_grouped

#     python misEvaluator.py --node_counts 10 15 20 25 30 --base_dir generated_graphs --output_dir mis_results_grouped
#     python misEvaluator.py --node_counts 45 50 --base_dir generated_graphs --output_dir mis_results_grouped_v2

