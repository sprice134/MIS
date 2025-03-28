import networkx as nx
import math
import random
import matplotlib.pyplot as plt

def create_bipartite_graph(n):
    # Ensure n is even
    if n % 2 != 0:
        raise ValueError("n must be even to divide equally into two groups.")
    
    # Define the two groups
    groupA = list(range(n // 2))
    groupB = list(range(n // 2, n))
    
    # Create an empty graph and add nodes with bipartite attribute
    G = nx.Graph()
    G.add_nodes_from(groupA, bipartite=0)
    G.add_nodes_from(groupB, bipartite=1)
    
    # Define the edge probability, p = (5*log(n))/n (ensuring p ≤ 1)
    p = min(5 * math.log(n) / n, 1)
    
    # Add edges between every node in groupA and every node in groupB with probability p
    for a in groupA:
        for b in groupB:
            if random.random() < p:
                G.add_edge(a, b)
                
    return G, groupA, groupB

def shuffle_node_ids(G):
    """Shuffle node ids while preserving node attributes, including the bipartite label."""
    original_nodes = list(G.nodes())
    shuffled_nodes = original_nodes[:]  # Copy list of nodes
    random.shuffle(shuffled_nodes)
    mapping = {old: new for old, new in zip(original_nodes, shuffled_nodes)}
    new_G = nx.relabel_nodes(G, mapping)
    
    # Recompute the groups based on the bipartite attribute
    new_groupA = [node for node, data in new_G.nodes(data=True) if data.get("bipartite") == 0]
    new_groupB = [node for node, data in new_G.nodes(data=True) if data.get("bipartite") == 1]
    
    return new_G, new_groupA, new_groupB

def compute_minimum_vertex_cover(G, top_nodes, bottom_nodes):
    """
    Compute the minimum vertex cover for a bipartite graph G using the 
    maximum matching (via Hopcroft-Karp algorithm) and the alternating path method.
    
    Returns:
      A set of nodes that form a minimum vertex cover.
    """
    # Compute maximum matching. The matching returned is a dictionary where if (u,v) is matched,
    # both matching[u]==v and matching[v]==u hold.
    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G, top_nodes)
    
    # U: Unmatched vertices in the top set
    unmatched = set(u for u in top_nodes if u not in matching)
    
    # Z: vertices reachable from unmatched vertices via alternating paths
    Z = set(unmatched)
    # We'll do a BFS on the "alternating" graph:
    queue = list(unmatched)
    while queue:
        u = queue.pop(0)
        if u in top_nodes:
            # For each neighbor v that is not in Z and where edge (u,v) is not in the matching:
            for v in G.neighbors(u):
                if v not in Z and matching.get(u) != v:
                    Z.add(v)
                    queue.append(v)
        else:
            # u is in bottom set: only follow edges that are in the matching
            for v in G.neighbors(u):
                if v not in Z and matching.get(u) == v:
                    Z.add(v)
                    queue.append(v)
    
    # According to the algorithm:
    # Minimum vertex cover = (top_nodes - Z) ∪ (bottom_nodes ∩ Z)
    min_vertex_cover = (set(top_nodes) - Z) | (set(bottom_nodes) & Z)
    return min_vertex_cover

def compute_ground_truth_mis(G, groupA, groupB):
    # Compute minimum vertex cover using our custom function
    min_vertex_cover = compute_minimum_vertex_cover(G, groupA, groupB)
    # Maximum independent set is the complement of the vertex cover
    all_nodes = set(G.nodes())
    mis = all_nodes - min_vertex_cover
    return mis

def greedy_min_degree_mis(G):
    # Make a copy of the graph so that we can remove nodes
    H = G.copy()
    mis = set()
    while H.nodes():
        # Select the node with the minimum degree
        node = min(H.nodes(), key=lambda n: H.degree(n))
        mis.add(node)
        # Remove the node and all its neighbors from the graph
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return mis

# Example usage:
n = 15000  # Ensure n is even
G, groupA, groupB = create_bipartite_graph(n)

# Shuffle the node IDs to avoid any bias from the original ordering
G, groupA, groupB = shuffle_node_ids(G)

# Compute the ground truth MIS via the computed minimum vertex cover
ground_truth_mis = compute_ground_truth_mis(G, groupA, groupB)
print("Ground Truth Maximum Independent Set:")
# print(ground_truth_mis)
print(len(ground_truth_mis))
# Compute the greedy min-degree MIS
greedy_mis = greedy_min_degree_mis(G)
print("\nGreedy Minimum-Degree MIS:")
# print(greedy_mis)
print(len(greedy_mis))