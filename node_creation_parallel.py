import os
import random
import networkx as nx
from multiprocessing import Pool, cpu_count

def generate_and_save_graph(params):
    """Worker function to generate and save a single graph based on provided parameters."""
    n, percent, iteration, base_dir = params
    
    # Create directory structure if not exists
    node_dir = os.path.join(base_dir, f"nodes_{n}")
    percent_dir = os.path.join(node_dir, f"removal_{percent}percent")
    os.makedirs(percent_dir, exist_ok=True)
    
    # Generate a complete graph with n nodes
    complete_graph = nx.complete_graph(n)
    all_edges = list(complete_graph.edges())
    total_edges = len(all_edges)
    
    # Determine how many edges to remove based on percent
    num_remove = int(total_edges * (percent / 100.0))
    
    # Copy complete graph and remove edges
    G = complete_graph.copy()
    edges_to_remove = random.sample(all_edges, num_remove)
    G.remove_edges_from(edges_to_remove)
    
    # Save graph to file
    filename = os.path.join(percent_dir, f"graph_{iteration}.edgelist")
    nx.write_edgelist(G, filename, data=False)
    
    # Return a status message (optional)
    return f"Saved graph: Nodes={n}, Removal={percent}%, Iteration={iteration}"

def gather_generation_tasks(node_counts, removal_percents, iterations, base_dir):
    """Prepare a list of tasks for generating graphs."""
    tasks = []
    for n in node_counts:
        for percent in removal_percents:
            for iteration in range(1, iterations + 1):
                tasks.append((n, percent, iteration, base_dir))
    return tasks

if __name__ == "__main__":
    # Define parameters
    node_counts = list(range(15, 155, 5))       # 15, 20, ..., 150
    removal_percents = list(range(15, 90, 5))  # 15%, 20%, ..., 85%
    iterations = 50
    base_dir = "generated_graphs"

    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Gather tasks
    tasks = gather_generation_tasks(node_counts, removal_percents, iterations, base_dir)
    print(f"Total graphs to generate: {len(tasks)}")

    # Determine the number of processes to use
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel graph generation.")

    # Use a pool of workers to generate graphs in parallel
    with Pool(processes=num_processes) as pool:
        for status in pool.imap_unordered(generate_and_save_graph, tasks):
            print(status)

    print("Graph generation complete.")
