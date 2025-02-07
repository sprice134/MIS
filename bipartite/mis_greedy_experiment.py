#!/usr/bin/env python3
"""
mis_greedy_experiment.py

This script evaluates two greedy algorithms for approximating a maximum independent set (MIS)
on two types of graphs:
  1. Bipartite random graphs: Two parts of equal size n with every cross–edge present with probability p.
     We vary p over 15 values starting at 5/1000 and increasing in arithmetic progression by 3/1000.
  2. Square grid graphs: N x N grid.

The two algorithms are:
  - Min–degree greedy: At each step, choose the node with the minimum degree.
  - Randomized greedy: At each step, choose a random node.

Results (average independent set size, and timing if desired) are printed for each experiment.
"""

import networkx as nx
import random
import numpy as np
import time

# ------------------------------
# Graph generation functions
# ------------------------------

def generate_bipartite_random_graph(n, p):
    """
    Generate a bipartite random graph with two parts of size n each.
    Every possible edge between the two parts is present with probability p.
    Returns a NetworkX graph.
    """
    B = nx.Graph()
    # Create nodes with bipartite attribute 0 or 1.
    left = range(0, n)
    right = range(n, 2*n)
    B.add_nodes_from(left, bipartite=0)
    B.add_nodes_from(right, bipartite=1)
    
    # Add cross edges with probability p.
    for u in left:
        for v in right:
            if random.random() < p:
                B.add_edge(u, v)
    return B

def generate_square_grid_graph(N):
    """
    Generate a square grid graph of size N x N.
    Returns a NetworkX graph.
    """
    # grid_2d_graph produces nodes as (i,j) tuples.
    G = nx.grid_2d_graph(N, N)
    # Optionally, relabel nodes to integers if desired.
    # Here we keep them as (i,j).
    return G

# ------------------------------
# Greedy MIS Algorithms
# ------------------------------

def min_degree_greedy(G):
    """
    Implements a greedy MIS algorithm that always picks a vertex of minimum degree.
    Returns the set of chosen vertices.
    """
    # Work on a copy of G.
    H = G.copy()
    mis = set()
    while H.number_of_nodes() > 0:
        # Choose node with minimum degree (ties broken arbitrarily)
        degrees = H.degree()
        u = min(H.nodes(), key=lambda x: degrees[x])
        mis.add(u)
        # Remove u and all its neighbors.
        neighbors = list(H.neighbors(u))
        H.remove_node(u)
        H.remove_nodes_from(neighbors)
    return mis

def randomized_greedy(G):
    """
    Implements a randomized greedy MIS algorithm.
    Returns the set of chosen vertices.
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
# Experiment functions
# ------------------------------

def experiment_bipartite(n, p, num_trials=10):
    """
    For a bipartite random graph with parts of size n and edge probability p,
    run num_trials experiments (each with a newly generated graph) using both
    min_degree_greedy and randomized_greedy. Return average MIS sizes.
    """
    sizes_min = []
    sizes_rand = []
    for _ in range(num_trials):
        G = generate_bipartite_random_graph(n, p)
        mis_min = min_degree_greedy(G)
        mis_rand = randomized_greedy(G)
        sizes_min.append(len(mis_min))
        sizes_rand.append(len(mis_rand))
    return np.mean(sizes_min), np.mean(sizes_rand)

def experiment_grid(N, num_trials=10):
    """
    For a square grid graph of size N x N, run num_trials experiments (using the same graph
    but different randomized runs for randomized_greedy) and return MIS sizes.
    For min_degree_greedy, the output is deterministic.
    """
    G = generate_square_grid_graph(N)
    size_min = len(min_degree_greedy(G))
    sizes_rand = []
    for _ in range(num_trials):
        mis_rand = randomized_greedy(G)
        sizes_rand.append(len(mis_rand))
    return size_min, np.mean(sizes_rand)

# ------------------------------
# Main experiment loop
# ------------------------------

def main():
    # Experiment parameters
    bipartite_n = 100      # each part has 100 nodes (adjust as needed)
    grid_N = 20            # grid graph of size 20x20 (adjust as needed)
    num_trials = 10
    
    # p values: 15 numbers starting at 5/1000 with gaps of 3/1000:
    p_values = [(5 + 3*i)/1000.0 for i in range(15)]
    
    print("=== Bipartite Random Graph Experiments ===")
    print("p\tMinDegree_MIS\tRandomized_MIS")
    for p in p_values:
        avg_min, avg_rand = experiment_bipartite(bipartite_n, p, num_trials)
        print(f"{p:.3f}\t{avg_min:.2f}\t\t{avg_rand:.2f}")
    
    print("\n=== Square Grid Graph Experiment ===")
    # For grid graphs we use a fixed graph and average the randomized result.
    min_size, avg_rand_grid = experiment_grid(grid_N, num_trials)
    print(f"Grid {grid_N}x{grid_N}:")
    print(f"MinDegree Greedy MIS size: {min_size}")
    print(f"Randomized Greedy average MIS size over {num_trials} trials: {avg_rand_grid:.2f}")

if __name__ == "__main__":
    main()
