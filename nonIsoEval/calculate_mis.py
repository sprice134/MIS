#!/usr/bin/env python3

import os
import json
import networkx as nx
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

def load_graph(file_path, n=5):
    """Load a graph from an edge list file and include all nodes."""
    G = nx.read_edgelist(file_path, nodetype=int)
    # Add all nodes to ensure isolated nodes are included
    G.add_nodes_from(range(n))
    return G

def mis_bruteforce(G):
    """
    Recursively find a maximum independent set of graph G.
    Warning: Exponential time complexity.
    """
    if len(G.nodes()) == 0:
        return []
    # Choose a node
    v = next(iter(G.nodes()))
    
    # Branch 1: Exclude v
    G_exclude = G.copy()
    G_exclude.remove_node(v)
    mis_exclude = mis_bruteforce(G_exclude)
    
    # Branch 2: Include v (and remove its neighbors)
    G_include = G.copy()
    neighbors = list(G_include.neighbors(v))
    G_include.remove_node(v)
    G_include.remove_nodes_from(neighbors)
    mis_include = [v] + mis_bruteforce(G_include)
    
    # Return whichever is larger
    return mis_exclude if len(mis_exclude) > len(mis_include) else mis_include

def process_graph(file_path, n=5):
    """
    Worker function to process a single graph file
    and return its MIS size plus the set of nodes in that MIS.
    """
    try:
        G = load_graph(file_path, n=n)
        mis_set = mis_bruteforce(G)
        mis_size = len(mis_set)
        print(f"Processed {file_path}: MIS size = {mis_size}")
        return {
            "file_path": file_path,
            "mis_size": mis_size,
            "maximum_independent_set": mis_set
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "mis_size": None,
            "maximum_independent_set": []
        }

def main():
    parser = argparse.ArgumentParser(
        description="Compute Maximum Independent Sets (MIS) from all .edgelist files in a directory."
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
        help="Directory containing all .edgelist files. If not specified, defaults to 'noniso_{n}_networkx'"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Name of the JSON file to store the results. If not specified, defaults to 'mis_results_{n}.json'"
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

    # Use all available CPU cores for parallel processing
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
