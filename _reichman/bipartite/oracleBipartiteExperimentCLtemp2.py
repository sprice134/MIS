import sys
import networkx as nx
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get node count from command line argument.
if len(sys.argv) < 2:
    print("Usage: python script.py <node_count>")
    sys.exit(1)
try:
    node_count = int(sys.argv[1])
except ValueError:
    print("Node count must be an integer.")
    sys.exit(1)

def create_bipartite_graph(n, coeff):
    """
    Create a bipartite graph with n nodes (n/2 in each partition).
    An edge between a node in group A and a node in group B is added with probability 
    p = min(coeff * log(n)/n, 1).
    """
    if n % 2 != 0:
        raise ValueError("n must be even to divide equally into two groups.")
    
    groupA = list(range(n // 2))
    groupB = list(range(n // 2, n))
    
    G = nx.Graph()
    G.add_nodes_from(groupA, bipartite=0)
    G.add_nodes_from(groupB, bipartite=1)
    
    p = min(coeff * math.log(n) / n, 1)
    
    for a in groupA:
        for b in groupB:
            if random.random() < p:
                G.add_edge(a, b)
                
    return G, groupA, groupB

def shuffle_node_ids(G):
    """Shuffle node ids while preserving node attributes, including the bipartite label."""
    original_nodes = list(G.nodes())
    shuffled_nodes = original_nodes[:]  
    random.shuffle(shuffled_nodes)
    mapping = {old: new for old, new in zip(original_nodes, shuffled_nodes)}
    new_G = nx.relabel_nodes(G, mapping)
    
    new_groupA = [node for node, data in new_G.nodes(data=True) if data.get("bipartite") == 0]
    new_groupB = [node for node, data in new_G.nodes(data=True) if data.get("bipartite") == 1]
    
    return new_G, new_groupA, new_groupB

def compute_minimum_vertex_cover(G, top_nodes, bottom_nodes):
    """
    Compute the minimum vertex cover for a bipartite graph G using the 
    maximum matching (via Hopcroft-Karp algorithm) and the alternating path method.
    """
    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G, top_nodes)
    unmatched = set(u for u in top_nodes if u not in matching)
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
    min_vertex_cover = compute_minimum_vertex_cover(G, groupA, groupB)
    all_nodes = set(G.nodes())
    mis = all_nodes - min_vertex_cover
    return mis

def greedy_min_degree_mis(G):
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
      - If v is in I_star, then with probability 1/2 + epsilon, mark v as 1; else 0.
      - If v is not in I_star, then with probability 1/2 + epsilon, mark v as 0; else 1.
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
    # Use the node count from the command line.
    n_values = [node_count]
    coeff_values = list(range(1, 11))  # 1, 2, ..., 10
    epislons = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    num_trials = 5  # Adjust as needed; set to 1 for brevity here
    base_seed = 42
    csv_filename = f"bipartite_experiment_results_epislons_{node_count}.csv"
    
    for n in n_values:
        for coeff in coeff_values:
            for trial in range(num_trials):
                for epsilon in epislons:

                    # Check if the CSV exists and if it already has a row with the same Coefficient, Epsilon, Trial.
                    if os.path.exists(csv_filename):
                        existing_df = pd.read_csv(csv_filename)
                        if not existing_df[
                            (existing_df["Coefficient"] == coeff) & 
                            (existing_df["Epsilon"] == epsilon) & 
                            (existing_df["Trial"] == trial)
                        ].empty:
                            print(f"Skipping n={n}, coeff={coeff}, trial={trial}, epsilon={epsilon} as entry already exists.")
                            continue

                    # Adjust seed for reproducibility and variability.
                    seed = int(base_seed) * int(trial) + int(n) * int(epsilon)
                    random.seed(seed)
                    
                    try:
                        G, groupA, groupB = create_bipartite_graph(n, coeff)
                    except Exception as e:
                        print(f"Error for n={n}, coeff={coeff}, seed={seed}: {e}")
                        continue
                    

                    G, groupA, groupB = shuffle_node_ids(G)
                    
                    ground_truth_mis = compute_ground_truth_mis(G, groupA, groupB)
                    greedy_mis = greedy_min_degree_mis(G)
                    
                    # Use ground truth MIS as I_star for the oracle simulation.
                    I_star = ground_truth_mis
                    barI, oracle = simulate_noisy_mis_oracle(G, I_star, epsilon)
                    
                    # Compute the noisy degree for each vertex.
                    tilde_deg = {v: sum(1 for nb in G.neighbors(v) if nb in barI) for v in G.nodes()}
                    
                    # Compute threshold s(v). For demonstration, use Delta = average degree.
                    avg_degree = sum(dict(G.degree()).values()) / float(G.number_of_nodes())
                    Delta = avg_degree
                    s = {v: (0.5 - epsilon) * G.degree(v) + 4 * math.sqrt(math.log(Delta)) * (0.5 - epsilon) * math.sqrt(Delta)
                        for v in G.nodes()}
                    
                    S = {v for v in G.nodes() if tilde_deg[v] <= s[v]}
                    H = G.subgraph(S)
                    mis_noise = greedy_min_degree_mis(H)
                    
                    result = {
                        "Coefficient": coeff,
                        "Nodes": n,
                        "Trial": trial,
                        "Epsilon": epsilon,
                        "Seed": seed,
                        "GroundTruth_MIS": len(ground_truth_mis),
                        "Greedy_MIS": len(greedy_mis),
                        "Size_S": len(S),
                        "Oracle_MIS": len(mis_noise),
                        "Improvement": len(mis_noise) - len(greedy_mis)
                    }
                    
                    print(f"n={n}, coeff={coeff}, trial={trial}, GT MIS={len(ground_truth_mis)}, "
                          f"Greedy MIS={len(greedy_mis)}, |S|={len(S)}, Oracle MIS={len(mis_noise)}, Improvement={result['Improvement']}")
                    
                    # Append the result to the CSV file.
                    header = not os.path.exists(csv_filename)
                    df_result = pd.DataFrame([result])
                    with open(csv_filename, "a", newline="") as f:
                        df_result.to_csv(f, index=False, header=header)
    
    print("Results saved to", csv_filename)


