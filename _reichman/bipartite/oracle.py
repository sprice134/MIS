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
    queue = list(unmatched)
    while queue:
        u = queue.pop(0)
        if u in top_nodes:
            for v in G.neighbors(u):
                if v not in Z and matching.get(u) != v:
                    Z.add(v)
                    queue.append(v)
        else:
            for v in G.neighbors(u):
                if v not in Z and matching.get(u) == v:
                    Z.add(v)
                    queue.append(v)
    
    min_vertex_cover = (set(top_nodes) - Z) | (set(bottom_nodes) & Z)
    return min_vertex_cover

def compute_ground_truth_mis(G, groupA, groupB):
    # Compute minimum vertex cover using our custom function
    min_vertex_cover = compute_minimum_vertex_cover(G, groupA, groupB)
    # The maximum independent set is the complement of the vertex cover
    all_nodes = set(G.nodes())
    mis = all_nodes - min_vertex_cover
    return mis

def greedy_min_degree_mis(G):
    """
    Compute an independent set using a min-degree greedy algorithm.
    Repeatedly pick the vertex with minimum degree, add it to the independent set,
    and remove it and its neighbors.
    """
    H = G.copy()
    mis = set()
    while H.nodes():
        v = min(H.nodes(), key=lambda n: H.degree(n))
        mis.add(v)
        neighbors = list(H.neighbors(v))
        H.remove_node(v)
        H.remove_nodes_from(neighbors)
    return mis

def simulate_noisy_mis_oracle(G, I_star, epsilon):
    """
    Simulate a noisy MIS oracle.
    
    For each vertex v:
      - If v is in I_star, then with probability 1/2 + epsilon, the oracle marks v as 1, otherwise as 0.
      - If v is not in I_star, then with probability 1/2 + epsilon, the oracle marks v as 0, otherwise as 1.
        
    Returns a set of vertices (oracle output = 1) and the full oracle dictionary.
    """
    oracle = {}
    for v in G.nodes():
        if v in I_star:
            oracle[v] = 1 if random.random() < (0.5 + epsilon) else 0
        else:
            oracle[v] = 0 if random.random() < (0.5 + epsilon) else 1
    barI = {v for v, val in oracle.items() if val == 1}
    return barI, oracle

# ------------------ Main Experiment ------------------ #

if __name__ == "__main__":
    # Set parameters
    n = 15000  # Total number of nodes (must be even)
    epsilon = 0.25  # Persistent noise level for the oracle

    # 1. Create and shuffle the bipartite graph.
    G, groupA, groupB = create_bipartite_graph(n)
    G, groupA, groupB = shuffle_node_ids(G)
    
    # 2. Compute the ground truth MIS using the minimum vertex cover method.
    ground_truth_mis = compute_ground_truth_mis(G, groupA, groupB)
    print("Ground Truth MIS size:", len(ground_truth_mis))
    
    # 3. Compute the greedy (maximal) MIS on the entire graph.
    greedy_mis = greedy_min_degree_mis(G)
    print("Greedy MIS size:", len(greedy_mis))
    
    # 4. Compute the exact maximum independent set for the noisy oracle simulation.
    # (Here we use the ground truth MIS as I_star.)
    I_star = ground_truth_mis
    
    # 5. Simulate the noisy MIS oracle with epsilon=0.25.
    barI, oracle = simulate_noisy_mis_oracle(G, I_star, epsilon)
    
    # 6. For each vertex, compute the noisy degree: number of neighbors with oracle output 1.
    tilde_deg = {v: sum(1 for nb in G.neighbors(v) if nb in barI) for v in G.nodes()}
    
    # 7. Compute threshold s(v) for each vertex.
    # For a d-regular graph, Delta = d; here, note that the bipartite graph isn't necessarily regular.
    # For demonstration, we'll set Delta = average degree.
    avg_degree = sum(dict(G.degree()).values()) / float(G.number_of_nodes())
    Delta = avg_degree
    s = {v: (0.5 - epsilon) * G.degree(v) + 4 * math.sqrt(math.log(Delta)) * (0.5 - epsilon) * math.sqrt(Delta)
         for v in G.nodes()}
    
    # 8. Form the set S = {v in V: tilde_deg(v) <= s(v)}
    S = {v for v in G.nodes() if tilde_deg[v] <= s[v]}
    print("Size of set S:", len(S))
    
    # 9. Run the min-degree greedy MIS on the induced subgraph G[S].
    H = G.subgraph(S)
    mis_noise = greedy_min_degree_mis(H)
    print("Oracle-driven MIS size:", len(mis_noise))
    
    # Optionally, you can visualize a small subgraph or print some statistics.
    # (Visualization is omitted here due to large n.)
