#!/usr/bin/env python3
"""
misEvaluator_exact.py

This script loads saved bipartite graph files (created by your generator) and computes the true 
maximum independent set (MIS) exactly using a maximum matching–based approach (via Kőnig’s theorem), 
which is efficient for bipartite graphs.

It also computes the MIS using two greedy algorithms for comparison:
  - min_degree_greedy: repeatedly picks the node with minimum degree.
  - randomized_greedy: repeatedly picks a random node.

The results for each graph (including node count, percent removed, MIS size, and the MIS sets) are 
saved into JSON files grouped by node count and removal percentage.

Assumed folder structure for graphs:
  <base_dir>/
      nodes_<n>/
          removal_<percent>percent/
              graph_<iteration>.edgelist

Example usage:
  python misEvaluator_exact.py --node_counts 10 15 20 25 --removal_percents 15 20 25 --iterations 50 --base_dir generated_graphs --output_dir mis_results_grouped
"""

import os
import json
import networkx as nx
import random
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import argparse
import numpy as np

# ------------------------------
# Graph Loading Function
# ------------------------------
def load_graph(file_path):
    edges = []
    nodes = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                u, v = map(int, parts)
                edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
            elif len(parts) == 1:
                node_id = int(parts[0])
                nodes.add(node_id)
            else:
                print(f"Warning: ignoring malformed line in {file_path}: {line}")
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (u, v) in edges:
        G.add_edge(u, v)
    return G


# ------------------------------
# Exact MIS for Bipartite Graphs (via Maximum Matching)
# ------------------------------
def exact_bipartite_mis(G, top_nodes=None):
    """
    Compute the maximum independent set exactly for bipartite graph G using Kőnig’s theorem.
    
    If top_nodes (one bipartition set) is not provided, compute the bipartite partition.
    Then:
      1. Compute a maximum matching.
      2. Let X = top_nodes and Y = V \ X.
      3. Let U be the unmatched vertices in X.
      4. Find all vertices reachable from U via alternating paths (from X use non–matching edges, 
         from Y use matching edges).
      5. The minimum vertex cover is (X \ Z) ∪ (Y ∩ Z).
      6. The maximum independent set is the complement: MIS = V \ (minimum vertex cover).
    
    Returns the set of nodes in the maximum independent set.
    """
    # If top_nodes not given, compute bipartite sets.
    if top_nodes is None:
        try:
            top_nodes, bottom_nodes = nx.bipartite.sets(G)
        except Exception as e:
            raise ValueError("Graph is not bipartite or cannot be partitioned.") from e
    else:
        bottom_nodes = set(G.nodes()) - set(top_nodes)
        
    # Compute maximum matching. (Matching dict has both directions.)
    matching = nx.bipartite.maximum_matching(G, top_nodes=top_nodes)
    
    # Unmatched vertices in X
    unmatched = [u for u in top_nodes if u not in matching]
    
    # Find all vertices reachable from unmatched vertices in X via alternating paths.
    Z = set(unmatched)
    stack = list(unmatched)
    while stack:
        u = stack.pop()
        if u in top_nodes:
            # From X, follow edges NOT in matching.
            for v in G.neighbors(u):
                if matching.get(u) != v and v not in Z:
                    Z.add(v)
                    stack.append(v)
        else:
            # From Y, follow the matching edge (if any).
            for v in G.neighbors(u):
                if v in top_nodes and matching.get(v) == u and v not in Z:
                    Z.add(v)
                    stack.append(v)
                    
    vertex_cover = (set(top_nodes) - Z) | (set(bottom_nodes) & Z)
    mis = set(G.nodes()) - vertex_cover
    return mis

# ------------------------------
# Greedy MIS Algorithms
# ------------------------------
def min_degree_greedy(G):
    """
    Greedy algorithm that repeatedly selects the node with minimum degree,
    adds it to the independent set, and removes it and its neighbors.
    Returns a set of nodes (the approximate MIS).
    """
    H = G.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        u = min(H.nodes(), key=lambda x: H.degree(x))
        mis.add(u)
        neighbors = list(H.neighbors(u))
        H.remove_node(u)
        H.remove_nodes_from(neighbors)
    return mis

def randomized_greedy(G):
    """
    Randomized greedy algorithm for MIS:
    Repeatedly pick a random node, add it to the independent set, and remove it and its neighbors.
    Returns a set of nodes (the approximate MIS).
    """
    H = G.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        u = random.choice(list(H.nodes()))
        mis.add(u)
        neighbors = list(H.neighbors(u))
        H.remove_node(u)
        H.remove_nodes_from(neighbors)
    return mis

# ------------------------------
# Process a Single Graph File
# ------------------------------
def process_graph(args):
    """
    Worker function to process a single graph file.
    Loads the graph, computes:
      - Exact MIS (using exact_bipartite_mis)
      - MIS from min_degree_greedy
      - MIS from randomized_greedy
    Packages the results into a dict for JSON saving.
    """
    file_path, n, percent, iteration = args
    try:
        G = load_graph(file_path)
        try:
            top_nodes, _ = nx.bipartite.sets(G)
        except Exception as e:
            print(f"Graph {file_path} is not bipartite; skipping.")
            return {
                "file_path": file_path,
                "num_nodes": n,
                "percent_removed": percent,
                "iteration": iteration,
                "MIS_SIZE_exact": None,
                "MIS_SETS_exact": [],
                "MIS_CELLS_exact": [],
                "MIS_CELLS_PROB_exact": [],
                "MIS_SIZE_min": None,
                "MIS_SIZE_rand": None
            }
        mis_exact = exact_bipartite_mis(G, top_nodes=top_nodes)
        mis_min = min_degree_greedy(G)
        mis_rand = randomized_greedy(G)
        
        # For exact MIS, we report:
        mis_size_exact = len(mis_exact)
        # Build a binary list for nodes 0..n-1: 1 if node is in MIS, else 0.
        mis_cells_exact = [1 if i in mis_exact else 0 for i in range(n)]
        # For probability, in an exact method there is exactly one MIS, so probability is 1 if in MIS, else 0.
        mis_cells_prob_exact = [float(val) for val in mis_cells_exact]
        
        result = {
            "file_path": file_path,
            "num_nodes": n,
            "percent_removed": percent,
            "iteration": iteration,
            "MIS_SIZE_exact": mis_size_exact,
            "MIS_SETS_exact": [sorted(list(mis_exact))],  # only one exact MIS (up to symmetry)
            "MIS_CELLS_exact": mis_cells_exact,
            "MIS_CELLS_PROB_exact": mis_cells_prob_exact,
            "MIS_SIZE_min": len(mis_min),
            "MIS_SIZE_rand": len(mis_rand)
        }
        print(f"Processed {file_path}: Exact MIS size = {mis_size_exact}, min_deg = {len(mis_min)}, rand = {len(mis_rand)}")
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "num_nodes": n,
            "percent_removed": percent,
            "iteration": iteration,
            "MIS_SIZE_exact": None,
            "MIS_SETS_exact": [],
            "MIS_CELLS_exact": [],
            "MIS_CELLS_PROB_exact": [],
            "MIS_SIZE_min": None,
            "MIS_SIZE_rand": None
        }

# ------------------------------
# Gather Tasks from Directory Structure
# ------------------------------
def gather_tasks(base_dir, node_counts_list, removal_percents, iterations):
    """
    Traverse the directory structure under base_dir to gather tasks.
    Assumes folders are named "nodes_<n>" and within them "removal_<percent>percent".
    Returns a list of tuples (file_path, n, percent, iteration).
    """
    tasks = []
    for node_folder in os.listdir(base_dir):
        node_folder_path = os.path.join(base_dir, node_folder)
        if not os.path.isdir(node_folder_path):
            continue

        # Extract node count from folder name "nodes_{n}"
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

            # Extract removal percent from folder name "removal_{percent}percent"
            try:
                percent_str = percent_folder.split('_')[1]
                percent = int(percent_str.replace('permil', ''))

            except (IndexError, ValueError):
                print(f"Warning: Unable to extract removal percent from folder '{percent_folder}'. Skipping.")
                continue

            if percent not in removal_percents:
                continue

            for graph_file in os.listdir(percent_folder_path):
                if not graph_file.endswith(".edgelist"):
                    continue
                file_path = os.path.join(percent_folder_path, graph_file)
                # Extract iteration from filename "graph_{iteration}.edgelist"
                try:
                    iteration_str = graph_file.split('_')[1]
                    iteration = int(iteration_str.split('.')[0])
                except (IndexError, ValueError):
                    print(f"Warning: Unable to extract iteration from file '{graph_file}'. Assigning iteration as None.")
                    iteration = None
                tasks.append((file_path, n, percent, iteration))
    return tasks

# ------------------------------
# Save Grouped Results to JSON
# ------------------------------
def save_grouped_results(results, output_dir):
    """
    Group results by node count and removal percent, then save each group as a JSON file.
    Files will be named "nodes_<n>_removal_<percent>percent.json" and saved to output_dir.
    """
    grouped = defaultdict(lambda: defaultdict(list))  # structure: {n: {percent: [results]}}
    for result in results:
        n = result["num_nodes"]
        percent = result["percent_removed"]
        grouped[n][percent].append(result)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for n, percent_dict in grouped.items():
        for percent, res_list in percent_dict.items():
            json_filename = f"nodes_{n}_removal_{percent}percent.json"
            json_path = os.path.join(output_dir, json_filename)
            try:
                with open(json_path, "w") as f:
                    json.dump(res_list, f, indent=2)
                print(f"Saved results for Nodes={n}, Removal={percent}% to '{json_path}'")
            except Exception as e:
                print(f"Error saving JSON file '{json_path}': {e}")

# ------------------------------
# Main Function
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute the exact Maximum Independent Set (MIS) for bipartite graphs (using maximum matching) and compare with greedy approximations."
    )
    parser.add_argument("--node_counts", type=int, nargs='+', required=True,
                        help="List of node counts to process, e.g., --node_counts 10 15 20")
    parser.add_argument("--removal_percents", type=int, nargs='+', default=list(range(15, 90, 5)),
                        help="List of edge removal percentages to process, e.g., --removal_percents 15 20 25")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations per combination of node count and removal percent")
    parser.add_argument("--base_dir", type=str, default="generated_graphs",
                        help="Base directory where generated graphs are stored")
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped",
                        help="Directory to save the grouped JSON result files")
    args = parser.parse_args()

    node_counts_list = args.node_counts
    removal_percents = args.removal_percents
    iterations = args.iterations
    base_dir = args.base_dir
    output_dir = args.output_dir

    print(f"Node counts to process: {node_counts_list}")
    print(f"Removal percentages to process: {removal_percents}%")
    print(f"Iterations per combination: {iterations}")
    print(f"Base directory: '{base_dir}'")
    print(f"Output directory: '{output_dir}'")

    tasks = gather_tasks(base_dir, node_counts_list, removal_percents, iterations)
    print(f"Total tasks to process: {len(tasks)}")
    if not tasks:
        print("No tasks found. Please check your directory structure and input parameters.")
        return

    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes.")

    results = []
    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(process_graph, tasks):
            results.append(result)
    
    print(f"Processed {len(results)} graphs.")
    save_grouped_results(results, output_dir)
    print("All grouped results saved.")

if __name__ == "__main__":
    main()
