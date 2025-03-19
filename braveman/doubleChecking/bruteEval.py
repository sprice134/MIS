#!/usr/bin/env python3
import sys
import itertools
import networkx as nx

def load_graph_from_edgelist(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            # If a line contains a single token, add it as an isolated node.
            if len(tokens) == 1:
                node = int(tokens[0])
                G.add_node(node)
            # Otherwise, assume first two tokens form an edge.
            elif len(tokens) >= 2:
                u, v = map(int, tokens[:2])
                G.add_edge(u, v)
                # Ensure both nodes are added.
                G.add_node(u)
                G.add_node(v)
    return G

def is_independent_set(G, subset):
    for u, v in G.edges():
        if u in subset and v in subset:
            return False
    return True

def brute_force_mis(G):
    nodes = list(G.nodes())
    best_set = set()
    best_size = 0
    n = len(nodes)
    # Enumerate all possible subsets using bitmask.
    for bitmask in range(1 << n):
        subset = {nodes[i] for i in range(n) if bitmask & (1 << i)}
        if is_independent_set(G, subset) and len(subset) > best_size:
            best_size = len(subset)
            best_set = subset
    return best_set

def greedy_mis(G):
    # Create a copy of the graph to work on.
    H = G.copy()
    independent_set = set()
    # While there are nodes remaining.
    while H.number_of_nodes() > 0:
        # Pick the node with the smallest degree (tie-break by node id).
        node = min(H.nodes(), key=lambda n: (H.degree(n), n))
        independent_set.add(node)
        # Remove the node and its neighbors.
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return independent_set

def main():
    if len(sys.argv) < 2:
        print("Usage: python mis_evaluation.py <edgelist_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    G = load_graph_from_edgelist(file_path)
    
    # Compute the brute-force maximum independent set.
    mis_brute = brute_force_mis(G)
    # Compute a greedy independent set.
    mis_greedy = greedy_mis(G)
    
    print("Brute Force Maximum Independent Set:")
    print("  Size =", len(mis_brute))
    print("  Nodes =", sorted(mis_brute))
    print("\nGreedy Independent Set:")
    print("  Size =", len(mis_greedy))
    print("  Nodes =", sorted(mis_greedy))

if __name__ == "__main__":
    main()
    '''
    python bruteEval.py graph_21.edgelist
    '''
