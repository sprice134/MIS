#!/usr/bin/env python3
"""
run_greedy_experiments.py

This script loads bipartite random graphs (previously generated and saved) and
evaluates two greedy algorithms for approximating a maximum independent set (MIS):
  - Min–degree greedy
  - Randomized greedy

It also runs the same experiments on a square grid graph.

Assumed folder structure for bipartite graphs:
  generated_graphs_bipartite/
      nodes_<n>/
          bipartite_<p_numer>permil/
              graph_<iteration>.edgelist

For bipartite graphs, total node counts are 100, 200, …, 1000 and for probabilities,
we consider 15 values: p = (5 + 3*i)/1000 for i = 0,1,…,14.
"""

import os
import random
import networkx as nx
import numpy as np

# ------------------------------
# Graph Loading Function
# ------------------------------

def load_graph_from_edgelist(filename):
    """
    Load a graph from an edgelist file.
    The file is expected to have lines either with two integers (edge: "u v")
    or a single integer (isolated node).
    Returns a NetworkX Graph.
    """
    G = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 2:
                u, v = tokens
                # Convert to int
                u = int(u)
                v = int(v)
                G.add_edge(u, v)
            elif len(tokens) == 1 and tokens[0] != "":
                u = int(tokens[0])
                G.add_node(u)
    return G

# ------------------------------
# Greedy MIS Algorithms
# ------------------------------

def min_degree_greedy(G):
    """
    Greedy MIS: repeatedly choose the node with the minimum degree.
    Returns a set of nodes in the independent set.
    """
    H = G.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        # Choose node with minimum degree (ties arbitrary)
        u = min(H.nodes(), key=lambda x: H.degree(x))
        mis.add(u)
        # Remove u and its neighbors
        neighbors = list(H.neighbors(u))
        H.remove_node(u)
        H.remove_nodes_from(neighbors)
    return mis

def randomized_greedy(G):
    """
    Randomized greedy MIS: repeatedly pick a random node.
    Returns a set of nodes in the independent set.
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
# Experiment for Bipartite Graphs (loaded from disk)
# ------------------------------

def experiment_loaded_bipartite(n, p_numer, base_dir, verbose=False):
    """
    For a given total node count n and probability numerator p_numer,
    load all graphs from:
      base_dir/nodes_<n>/bipartite_<p_numer>permil/
    and run both greedy algorithms.
    Returns a tuple (avg_mis_size_min, avg_mis_size_rand, num_graphs).
    """
    folder = os.path.join(base_dir, f"nodes_{n}", f"bipartite_{p_numer}permil")
    if not os.path.exists(folder):
        if verbose:
            print(f"Folder {folder} does not exist. Skipping.")
        return None, None, 0

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".edgelist")]
    if len(files) == 0:
        if verbose:
            print(f"No files found in {folder}.")
        return None, None, 0

    sizes_min = []
    sizes_rand = []
    for filename in files:
        G = load_graph_from_edgelist(filename)
        mis_min = min_degree_greedy(G)
        mis_rand = randomized_greedy(G)
        sizes_min.append(len(mis_min))
        sizes_rand.append(len(mis_rand))
    avg_min = np.mean(sizes_min)
    avg_rand = np.mean(sizes_rand)
    return avg_min, avg_rand, len(files)

# ------------------------------
# Experiment for Square Grid Graphs
# ------------------------------

def generate_square_grid_graph(N):
    """
    Generate a square grid graph of size N x N.
    Returns a NetworkX Graph.
    """
    G = nx.grid_2d_graph(N, N)
    # Optionally, relabel nodes as integers if desired.
    # For now, we keep the tuple labels.
    return G

def experiment_grid(N, num_trials=10):
    """
    Generate an N x N grid graph and run greedy experiments.
    Returns a tuple (mis_size_min, avg_mis_rand) where:
      - mis_size_min is the MIS size from min_degree_greedy (deterministic)
      - avg_mis_rand is the average MIS size from randomized_greedy over num_trials.
    """
    G = generate_square_grid_graph(N)
    mis_min = min_degree_greedy(G)
    size_min = len(mis_min)
    sizes_rand = []
    for _ in range(num_trials):
        mis_rand = randomized_greedy(G)
        sizes_rand.append(len(mis_rand))
    return size_min, np.mean(sizes_rand)

# ------------------------------
# Main Experiment Loop
# ------------------------------

def main():
    base_dir = "generated_graphs_bipartite"  # folder where bipartite graphs were saved
    # Total node counts: 100, 200, ..., 1000
    node_counts = list(range(100, 1001, 100))
    # p_numerators: 15 numbers starting at 5 with gap of 3: 5, 8, 11, ..., 47
    p_numerators = [5 + 3*i for i in range(15)]
    
    print("=== Bipartite Graph Experiments ===")
    print("Nodes\tp\tMinDeg_MIS\tRand_MIS")
    for n in node_counts:
        for p_num in p_numerators:
            avg_min, avg_rand, num_graphs = experiment_loaded_bipartite(n, p_num, base_dir)
            if num_graphs == 0:
                continue
            # p value for printing
            p_val = p_num / 1000.0
            print(f"{n}\t{p_val:.3f}\t{avg_min:.2f}\t\t{avg_rand:.2f}")
    
    print("\n=== Square Grid Graph Experiment ===")
    grid_size = 20  # Create a 20x20 grid graph
    num_trials = 10
    min_size, avg_rand_grid = experiment_grid(grid_size, num_trials)
    print(f"Grid {grid_size}x{grid_size}:")
    print(f"MinDegree Greedy MIS size: {min_size}")
    print(f"Randomized Greedy average MIS size over {num_trials} trials: {avg_rand_grid:.2f}")

if __name__ == "__main__":
    main()
