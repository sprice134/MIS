import os
import json
import networkx as nx
from multiprocessing import Pool, cpu_count
from collections import defaultdict

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

def process_graph_greedy(args):
    """Worker function to process a single graph file using the greedy algorithm."""
    file_path, n, percent, iteration = args
    try:
        G = load_graph(file_path)
        mis_set = greedy_mis(G)  # Use greedy_mis instead of mis_bruteforce
        mis_size = len(mis_set)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        mis_size = None
        mis_set = []
    
    result = {
        "file_path": file_path,
        "num_nodes": n,
        "percent_removed": percent,
        "iteration": iteration,
        "greedy_mis_size": mis_size,
        "greedy_maximum_independent_set": mis_set
    }
    print(f"Processed {file_path}: Greedy MIS size = {mis_size}")
    return result

def gather_tasks(base_dir, node_counts_list, removal_percents):
    """Traverse directory structure to gather tasks based on specified node counts."""
    tasks = []
    for node_folder in os.listdir(base_dir):
        node_folder_path = os.path.join(base_dir, node_folder)
        if not os.path.isdir(node_folder_path):
            continue

        # Extract node count from folder name
        try:
            n = int(node_folder.split('_')[1])
        except (IndexError, ValueError):
            continue

        if n not in node_counts_list:
            continue  # Skip node counts not in the specified list

        for percent_folder in os.listdir(node_folder_path):
            percent_folder_path = os.path.join(node_folder_path, percent_folder)
            if not os.path.isdir(percent_folder_path):
                continue

            # Extract percent removed from folder name
            try:
                percent = int(percent_folder.split('_')[1].replace('percent', ''))
            except (IndexError, ValueError):
                continue

            if percent not in removal_percents:
                continue  # Skip removal percents not in the specified list

            for graph_file in os.listdir(percent_folder_path):
                if not graph_file.endswith(".edgelist"):
                    continue

                file_path = os.path.join(percent_folder_path, graph_file)

                # Extract iteration from filename (assuming format "graph_X.edgelist")
                try:
                    iteration = int(graph_file.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    iteration = None

                tasks.append((file_path, n, percent, iteration))
    return tasks

def save_grouped_results(results, output_dir):
    """
    Group results by node count and percent removed,
    then save each group into a separate JSON file.
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
            json_filename = f"nodes_{n}_removal_{percent}percent_greedy.json"
            json_path = os.path.join(output_dir, json_filename)
            
            # Save the list of results to the JSON file
            with open(json_path, "w") as f:
                json.dump(res_list, f, indent=2)
            
            print(f"Saved greedy results for Nodes={n}, Removal={percent}% to {json_path}")

def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Parallel Greedy MIS Computation with Grouped JSON Outputs")
    parser.add_argument(
        "--node_counts",
        type=int,
        nargs='+',
        required=True,
        help="List of node counts to process"
    )
    parser.add_argument(
        "--removal_percents",
        type=int,
        nargs='+',
        default=list(range(15, 90, 5)),
        help="List of edge removal percentages to process"
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
        default="greedy_mis_results_grouped",
        help="Directory to save the grouped JSON result files for greedy MIS"
    )
    args = parser.parse_args()

    node_counts_list = args.node_counts
    removal_percents = args.removal_percents
    base_dir = args.base_dir
    output_dir = args.output_dir

    print(f"Node counts to process: {node_counts_list}")
    print(f"Removal percentages to process: {removal_percents}%")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")

    # Gather tasks based on specified node counts and removal percents
    tasks = gather_tasks(base_dir, node_counts_list, removal_percents)
    print(f"Total tasks to process: {len(tasks)}")

    if not tasks:
        print("No tasks found. Please check your directory structure and input parameters.")
        return

    # Determine the number of processes to use
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel computation.")

    results = []
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Map tasks to the pool. Using imap_unordered for efficiency and to handle results as they come in.
        for result in pool.imap_unordered(process_graph_greedy, tasks):
            results.append(result)

    print(f"All tasks processed. Total results collected: {len(results)}")

    # Save grouped results into separate JSON files
    save_grouped_results(results, output_dir)

    print("All grouped greedy results have been saved.")

if __name__ == "__main__":
    main()
