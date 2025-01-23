import os
import json
import networkx as nx
from multiprocessing import Pool, cpu_count
from functools import partial

def load_graph(file_path):
    """Load a graph from an edge list file."""
    return nx.read_edgelist(file_path, nodetype=int)

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

def process_graph(file_path):
    """
    Worker function to process a single graph file
    and return its MIS size plus the set of nodes in that MIS.
    """
    try:
        G = load_graph(file_path)
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Maximum Independent Sets (MIS) from all .edgelist files in a single directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="noniso_6_networkx",
        help="Directory containing all .edgelist files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="mis_results.json",
        help="Name of the JSON file to store the results"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

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
        for result in pool.imap_unordered(process_graph, edgelist_files):
            results.append(result)

    print(f"All tasks processed. Total results collected: {len(results)}")

    # Save all results in a single JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
