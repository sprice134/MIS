#!/usr/bin/env python3
"""
Optimized MIS evaluator for bipartite graphs

This script processes graph files (in edge list format) organized as:
  /home/sprice/MIS/bipartite/generated_graphs_bipartite/
      nodes_<n>/
          bipartite_<removal>permil/
              graph_<iteration>.edgelist

For each graph file:
  - The graph is loaded (isolated nodes are added).
  - Isolated (unattached) nodes are identified and removed.
  - The remaining graph is assumed to be bipartite and is processed via a maximum
    matching algorithm (using Kőnig’s theorem) to compute the maximum independent set.
  - The isolated nodes are then unioned with the computed MIS.
  - Per-node statistics are computed:
      * MIS_CELLS: 1 if the node appears in the MIS; else 0.
      * MIS_CELLS_PROB: Since there is a unique MIS (up to symmetry), probability is 1 for nodes in the MIS and 0 otherwise.
  - Results are grouped and saved as JSON files.
  
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
# Graph Loading
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

# -------------------------
# Exact MIS for Bipartite Graphs via Maximum Matching
# -------------------------

def exact_bipartite_mis(G, top_nodes):
    """
    Compute the maximum independent set exactly for a bipartite graph G using Kőnig’s theorem.
    
    Given a bipartition (top_nodes and Y = V \ top_nodes), the procedure is:
      1. Compute a maximum matching.
      2. Let U be the unmatched vertices in top_nodes.
      3. Compute Z = vertices reachable from U via alternating paths 
         (from X follow non–matching edges, from Y follow matching edges).
      4. Then the minimum vertex cover is (top_nodes - Z) ∪ (Y ∩ Z).
      5. The maximum independent set is the complement: MIS = V \ (minimum vertex cover).
    
    Returns the MIS as a set of nodes.
    """
    Y = set(G.nodes()) - set(top_nodes)
    matching = nx.bipartite.maximum_matching(G, top_nodes=top_nodes)
    # Get unmatched vertices in top_nodes
    unmatched = [u for u in top_nodes if u not in matching]
    Z = set(unmatched)
    stack = list(unmatched)
    while stack:
        u = stack.pop()
        if u in top_nodes:
            # From top, follow edges NOT in the matching.
            for v in G.neighbors(u):
                if matching.get(u) != v and v not in Z:
                    Z.add(v)
                    stack.append(v)
        else:
            # From Y, follow the matching edge.
            for v in G.neighbors(u):
                if v in top_nodes and matching.get(v) == u and v not in Z:
                    Z.add(v)
                    stack.append(v)
    vertex_cover = (set(top_nodes) - Z) | (Y & Z)
    mis = set(G.nodes()) - vertex_cover
    return mis

# -------------------------
# Process Single Graph (Worker)
# -------------------------

def process_graph(args):
    """Worker function to process a single graph file."""
    file_path, n, removal, iteration = args
    try:
        G = load_graph(file_path)
        # Identify isolated nodes (which can always be in the MIS)
        isolated = set(nx.isolates(G))
        
        # Remove isolated nodes before processing the bipartite part.
        if isolated:
            G_non_iso = G.copy()
            G_non_iso.remove_nodes_from(isolated)
        else:
            G_non_iso = G

        if G_non_iso.number_of_nodes() > 0:
            # Use a bipartite coloring to get the two sets.
            # nx.bipartite.color works even if the graph is disconnected.
            color = nx.algorithms.bipartite.color(G_non_iso)
            top_nodes = {node for node, col in color.items() if col == 0}
            # Compute the exact MIS for the bipartite graph.
            mis_non_iso = exact_bipartite_mis(G_non_iso, top_nodes)
        else:
            mis_non_iso = set()
        
        # Union isolated nodes with the computed MIS.
        mis = mis_non_iso.union(isolated)
        mis_size = len(mis)
        mis_list = sorted(list(mis))
        
        # Per-node statistics.
        # MIS_CELLS: 1 if node is in the MIS; 0 otherwise.
        mis_cells = [1 if i in mis else 0 for i in range(n)]
        # Since there's only one exact MIS, MIS_CELLS_PROB is 1 or 0.
        mis_cells_prob = [float(val) for val in mis_cells]
        
        result = {
            "file_path": file_path,
            "num_nodes": n,
            "removal": removal,
            "iteration": iteration,
            "MIS_SIZE": mis_size,
            "MIS_SET": mis_list,
            "MIS_CELLS": mis_cells,
            "MIS_CELLS_PROB": mis_cells_prob
        }
        print(f"Processed {file_path}: MIS size = {mis_size}")
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "num_nodes": n,
            "removal": removal,
            "iteration": iteration,
            "MIS_SIZE": None,
            "MIS_SET": [],
            "MIS_CELLS": [],
            "MIS_CELLS_PROB": []
        }

# -------------------------
# Task Gathering and Result Saving
# -------------------------

def gather_tasks(base_dir, node_counts_list, removal_values, iterations):
    """
    Traverse the directory structure to gather tasks.
    Expected structure:
      <base_dir>/nodes_<n>/bipartite_<removal>permil/
          graph_<iteration>.edgelist
    """
    tasks = []
    for node_folder in os.listdir(base_dir):
        node_folder_path = os.path.join(base_dir, node_folder)
        if not os.path.isdir(node_folder_path):
            continue

        # Extract node count from folder name (e.g., "nodes_100")
        try:
            n = int(node_folder.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Unable to extract node count from folder '{node_folder}'. Skipping.")
            continue

        if n not in node_counts_list:
            continue

        for subfolder in os.listdir(node_folder_path):
            subfolder_path = os.path.join(node_folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            try:
                # Expect folder name like "bipartite_5permil" or "bipartite_47permil"
                removal_str = subfolder.split('_')[1]
                removal = int(removal_str.replace('permil', ''))
            except (IndexError, ValueError):
                print(f"Warning: Unable to extract removal value from folder '{subfolder}'. Skipping.")
                continue

            if removal not in removal_values:
                continue

            for graph_file in os.listdir(subfolder_path):
                if not graph_file.endswith(".edgelist"):
                    continue
                file_path = os.path.join(subfolder_path, graph_file)
                try:
                    # Extract iteration number (assume filename "graph_<iteration>.edgelist")
                    iteration_str = graph_file.split('_')[1]
                    iteration = int(iteration_str.split('.')[0])
                except (IndexError, ValueError):
                    print(f"Warning: Unable to extract iteration from file '{graph_file}'. Assigning iteration as None.")
                    iteration = None

                tasks.append((file_path, n, removal, iteration))
    return tasks

def save_group_result(n, removal, results, output_dir):
    """
    Save results for a given node count and removal group into a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_filename = f"nodes_{n}_bipartite_{removal}permil.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for Nodes={n}, Removal={removal} permil to '{json_path}'")
    except Exception as e:
        print(f"Error saving JSON file '{json_path}': {e}")

# -------------------------
# Main function
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute maximum independent sets (MIS) for bipartite .edgelist graphs using a matching-based approach."
    )
    parser.add_argument("--node_counts", type=int, nargs='+', required=True,
                        help="List of node counts to process, e.g., --node_counts 100 1000")
    parser.add_argument("--removal_values", type=int, nargs='+', default=[5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47,
                                                                          50, 53, 56, 69, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92,
                                                                         95, 98, 101, 104, 107],
                        help="List of removal values to process (extracted from folder names), e.g., --removal_values 5 47")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations per combination")
    parser.add_argument("--base_dir", type=str, default="/home/sprice/MIS/bipartite/generated_graphs_bipartite",
                        help="Base directory where generated graphs are stored")
    parser.add_argument("--output_dir", type=str, default="mis_results_grouped",
                        help="Directory to save the grouped JSON result files")
    args = parser.parse_args()

    print(f"Processing node counts: {args.node_counts}")
    print(f"Processing removal values: {args.removal_values}")
    print(f"Iterations per combination: {args.iterations}")
    print(f"Base directory: '{args.base_dir}', Output directory: '{args.output_dir}'")
    
    tasks = gather_tasks(args.base_dir, args.node_counts, args.removal_values, args.iterations)
    print(f"Total tasks to process: {len(tasks)}")
    if not tasks:
        print("No tasks found. Check your directory structure and parameters.")
        return

    # Determine expected results per (n, removal) group.
    group_expected = defaultdict(int)
    for task in tasks:
        _, n, removal, _ = task
        group_expected[(n, removal)] += 1

    # Dictionary to accumulate results by group.
    group_results = defaultdict(list)

    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes.")
    
    total_results = 0
    with Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(process_graph, tasks):
            total_results += 1
            key = (res["num_nodes"], res["removal"])
            group_results[key].append(res)
            # When a group is complete, save its JSON file.
            if len(group_results[key]) == group_expected[key]:
                save_group_result(key[0], key[1], group_results[key], args.output_dir)
                del group_results[key]

    print(f"All tasks processed. Total results collected: {total_results}")
    
    # Save any remaining groups.
    if group_results:
        print("Saving any remaining groups...")
        for (n, removal), res_list in group_results.items():
            save_group_result(n, removal, res_list, args.output_dir)
    
    print("All grouped results have been saved.")

if __name__ == "__main__":
    main()

"""
Example usage:
    python mis_evaluator.py --node_counts 100 200 300 400 500 600 700 800 900 1000 --iterations 100 \
         --base_dir /home/sprice/MIS/bipartite/generated_graphs_bipartite \
         --output_dir mis_results_grouped
"""
