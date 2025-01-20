import os
import json
import networkx as nx
import random
from multiprocessing import Pool, cpu_count

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
    
    # Return the larger independent set
    return mis_exclude if len(mis_exclude) > len(mis_include) else mis_include

def process_graph(args):
    """Worker function to process a single graph file."""
    file_path, n, percent, iteration = args
    try:
        G = load_graph(file_path)
        mis_set = mis_bruteforce(G)
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
        "mis_size": mis_size,
        "maximum_independent_set": mis_set
    }
    print(f"Processed {file_path}: MIS size = {mis_size}")
    return result

def gather_tasks(base_dir):
    """Traverse directory structure to gather all tasks."""
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

        for percent_folder in os.listdir(node_folder_path):
            percent_folder_path = os.path.join(node_folder_path, percent_folder)
            if not os.path.isdir(percent_folder_path):
                continue

            # Extract percent removed from folder name
            try:
                percent = int(percent_folder.split('_')[1].replace('percent', ''))
            except (IndexError, ValueError):
                continue

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

if __name__ == "__main__":
    base_dir = "generated_graphs"
    output_file = "mis_results.json"

    # Gather all tasks from directory structure
    tasks = gather_tasks(base_dir)
    print(f"Total tasks to process: {len(tasks)}")

    # Determine the number of processes to use
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel computation.")

    results = []
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Map tasks to the pool. Using imap_unordered for efficiency and to handle results as they come in.
        for result in pool.imap_unordered(process_graph, tasks):
            results.append(result)

    # Save all results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"All results saved to {output_file}")
