import os
import argparse
import json
import networkx as nx
from itertools import permutations
from collections import Counter
from tqdm import tqdm
import math

def load_graph(file_path, n=5):
    """Load a graph from an edge list file and include all nodes."""
    G = nx.read_edgelist(file_path, nodetype=int)
    # Add all nodes to ensure isolated nodes are included (0..n-1)
    G.add_nodes_from(range(n))
    return G

def greedy_mis(G):
    """
    Greedy algorithm to find an independent set by repeatedly selecting
    the node with the fewest neighbors, breaking ties by the smallest label.
    """
    independent_set = []
    # Work on a copy of the graph to preserve the original
    H = G.copy()
    while H.nodes():
        # Select the node with minimum (degree, label).
        # This ensures we break ties on the label in the new labeling.
        node = min(H.nodes(), key=lambda x: (H.degree(x), x))
        
        independent_set.append(node)
        # Remove the chosen node and its neighbors from the graph
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return independent_set

def relabel_graph(G, mapping):
    """Relabel the graph according to the provided mapping (old->new)."""
    return nx.relabel_nodes(G, mapping)

def generate_all_isomorphic_graphs(G):
    """
    Generate all isomorphic copies of G by permuting node labels.
    
    :param G: NetworkX graph
    :return: List of isomorphic NetworkX graphs
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    all_perms = permutations(nodes)
    isomorphic_graphs = []
    
    for perm in tqdm(all_perms, desc="Generating isomorphic graphs", total=math.factorial(n)):
        mapping = {old: new for old, new in zip(nodes, perm)}
        G_iso = relabel_graph(G, mapping)
        isomorphic_graphs.append(G_iso)
    
    return isomorphic_graphs

def aggregate_greedy_mis_frequencies(G):
    """
    Generate all isomorphic copies of G, compute their greedy MIS sizes,
    and aggregate the frequencies of each MIS size.
    
    :param G: NetworkX graph
    :return: Counter object with MIS sizes as keys and frequencies as values
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    all_perms = permutations(nodes)
    mis_sizes = []
    
    for perm in tqdm(all_perms, desc="Processing permutations", total=math.factorial(n)):
        mapping = {old: new for old, new in zip(nodes, perm)}
        # Relabel G under this permutation
        G_iso = relabel_graph(G, mapping)
        # Run the (tie-break-corrected) greedy MIS
        mis_set = greedy_mis(G_iso)
        mis_size = len(mis_set)
        mis_sizes.append(mis_size)
    
    frequency = Counter(mis_sizes)
    return frequency

def main():
    parser = argparse.ArgumentParser(
        description="Generate all isomorphic copies of a graph, compute their Greedy MIS sizes (with proper tie-break), and aggregate frequencies."
    )
    parser.add_argument(
        "--graph_file",
        type=str,
        required=True,
        help="Path to the .edgelist file of the graph."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="Number of nodes in the graph (default: 5)"
    )
    args = parser.parse_args()

    graph_file = args.graph_file
    n = args.nodes

    if not os.path.isfile(graph_file):
        print(f"Error: The file '{graph_file}' does not exist.")
        return

    print(f"Loading graph from '{graph_file}' with {n} nodes...")
    G = load_graph(graph_file, n=n)

    print("Aggregating Greedy MIS frequencies across all isomorphic copies...")
    frequency = aggregate_greedy_mis_frequencies(G)

    print("\nAggregated Frequency of Greedy MIS Sizes:")
    for mis_size, count in sorted(frequency.items()):
        print(f"Greedy MIS Size {mis_size}: {count} occurrences")

if __name__ == "__main__":
    main()

    # python evaluate_iso_greedy.py --graph_file noniso_7_networkx/graph_666.edgelist --nodes 7