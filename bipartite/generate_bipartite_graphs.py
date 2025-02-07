#!/usr/bin/env python3
"""
generate_bipartite_graphs.py

This script generates and saves bipartite random graphs.

For each total node count in [100, 200, ..., 1000] and for each edge probability 
p = (5 + 3*i)/1000 for i = 0,...,14, it generates a number of graphs (iterations)
and saves each as an edge list file. The bipartite graph has two parts of equal size.

The output files are organized as follows:

    <base_dir>/nodes_<n>/bipartite_<p_numerators>permil/graph_<iteration>.edgelist

For example, a 100–node graph with p = 0.005 (numerator 5) is stored under
    generated_graphs/nodes_100/bipartite_5permil/graph_1.edgelist
"""

import os
import random
import networkx as nx
from multiprocessing import Pool, cpu_count

def generate_and_save_bipartite_graph(params):
    """
    Worker function to generate and save a single bipartite random graph.
    
    Parameters tuple contains:
        n         : Total number of nodes (must be even; here n/2 per part)
        p_numer   : Numerator for the probability p (i.e. p = p_numer/1000)
        iteration : Which iteration (for filename uniqueness)
        base_dir  : Base directory where graphs are saved
    """
    n, p_numer, iteration, base_dir = params
    p = p_numer / 1000.0

    # Determine the sizes of the two parts.
    # We assume n is even (or else use floor/ceil)
    n_left = n // 2
    n_right = n - n_left  # In case n is odd

    # Create directory structure
    node_dir = os.path.join(base_dir, f"nodes_{n}")
    # Folder name indicating bipartite graph with probability p (using the numerator)
    prob_dir = os.path.join(node_dir, f"bipartite_{p_numer}permil")
    os.makedirs(prob_dir, exist_ok=True)

    # Generate bipartite random graph.
    # networkx's bipartite_random_graph uses parameters: number of nodes in each set and p.
    B = nx.bipartite.random_graph(n_left, n_right, p, seed=random.randint(0, 10**6))
    
    # For consistency, relabel nodes: left nodes as 0,...,n_left-1 and right nodes as n_left,...,n-1.
    mapping = {}
    left_nodes = list(range(0, n_left))
    right_nodes = list(range(n_left, n))
    # The generated graph uses nodes labeled 0...n_left-1 and n_left...n_left+n_right-1.
    # We remap right nodes to start at n_left (if not already).
    for u in B.nodes():
        if u < n_left:
            mapping[u] = u
        else:
            mapping[u] = u - n_left + n_left  # essentially u remains u if already in that range
    B = nx.relabel_nodes(B, mapping)

    # Identify isolated nodes after edge generation
    isolated_nodes = list(nx.isolates(B))
    
    # Save graph to file: each edge as "u v" and isolated nodes as "u"
    filename = os.path.join(prob_dir, f"graph_{iteration}.edgelist")
    with open(filename, 'w') as f:
        # Write edges
        for u, v in B.edges():
            f.write(f"{u} {v}\n")
        # Write isolated nodes (if any)
        for u in isolated_nodes:
            f.write(f"{u}\n")
    
    return f"Saved bipartite graph: Nodes={n}, p={p:.3f} ({p_numer} permil), Iteration={iteration}"

def gather_bipartite_tasks(node_counts, p_numerators, iterations, base_dir):
    """
    Prepare a list of tasks for generating bipartite graphs.
    
    node_counts: list of total node counts (e.g., [100, 200, ..., 1000])
    p_numerators: list of numerators (e.g., [5, 8, 11, ..., ?])
    iterations: number of graphs to generate per parameter combination.
    base_dir: base directory for storing graphs.
    """
    tasks = []
    for n in node_counts:
        for p_numer in p_numerators:
            for iteration in range(1, iterations + 1):
                tasks.append((n, p_numer, iteration, base_dir))
    return tasks

if __name__ == "__main__":
    # Define parameters:
    # Total node counts: 100, 200, ..., 1000
    node_counts = list(range(100, 1001, 100))
    # p_numerators: 15 numbers starting at 5 with gap of 3:
    p_numerators = [5 + 3 * i for i in range(15)]
    # Number of graphs to generate per combination:
    iterations = 100
    # Base directory to store graphs:
    base_dir = "generated_graphs_bipartite"
    
    # Create the base directory if not exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Gather tasks
    tasks = gather_bipartite_tasks(node_counts, p_numerators, iterations, base_dir)
    total_tasks = len(tasks)
    print(f"Total bipartite graphs to generate: {total_tasks}")
    
    # Use multiprocessing pool
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel graph generation.")
    
    with Pool(processes=num_processes) as pool:
        for status in pool.imap_unordered(generate_and_save_bipartite_graph, tasks):
            print(status)
    
    print("Bipartite graph generation complete.")
