import os
import json
import networkx as nx
from multiprocessing import Pool, cpu_count

def load_graph(file_path):
    """Load a graph from an edge list file."""
    return nx.read_edgelist(file_path, nodetype=int)

def greedy_mis(G):
    """
    Greedy algorithm to find an independent set by repeatedly selecting
    the node with the fewest neighbors.
    """
    independent_set = []
    # Work on a copy of the graph to preserve the original
    H = G.copy()
    while H.nodes():
        # Select the node with the minimum degree
        node = min(H.nodes(), key=lambda n: H.degree(n))
        independent_set.append(node)
        # Remove the chosen node and its neighbors from the graph
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return independent_set

def process_graph_greedy(file_path):
    """
    Worker function to process a single graph file using the greedy algorithm.
    Returns a dict containing file_path and the MIS result.
    """
    try:
        G = load_graph(file_path)
        mis_set = greedy_mis(G)
        mis_size = len(mis_set)
        print(f"Processed {file_path}: Greedy MIS size = {mis_size}")
        return {
            "file_path": file_path,
            "greedy_mis_size": mis_size,
            "greedy_mis_set": mis_set
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "greedy_mis_size": None,
            "greedy_mis_set": []
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute a Greedy MIS for all .edgelist files in a single directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="generated_graphs",
        help="Directory containing all .edgelist files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="greedy_mis_results.json",
        help="Path to the JSON file to store the results"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return

    # Collect all .edgelist files in the given directory
    edgelist_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".edgelist")
    ]

    if not edgelist_files:
        print(f"No .edgelist files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(edgelist_files)} .edgelist files in '{input_dir}'.")

    # Use all available cores for parallel processing
    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes.")

    results = []
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Distribute the work; imap_unordered returns results as they become ready
        for result in pool.imap_unordered(process_graph_greedy, edgelist_files):
            results.append(result)

    print(f"All tasks processed. Collected results for {len(results)} graphs.")

    # Save everything into one JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Greedy MIS results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
