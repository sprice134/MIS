import os
import sys
import itertools
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from filelock import FileLock

# ===================== DOE Locking and Update Workflow =====================

# Set file names for DOE and lock.
doe_csv_filename = "experimentalDOE.csv"
lock_filename = doe_csv_filename + ".lock"
result_columns = ["GroundTruth_MIS", "Greedy_MIS", "Size_S", "Oracle_MIS", "Improvement"]

def get_next_experiment():
    """
    Opens the DOE file with a lock, finds the first row with empty result fields,
    sets those fields to -1 (marking it as reserved), saves the file, and returns 
    the row index along with its content as a dictionary.
    """
    with FileLock(lock_filename, timeout=10):
        df = pd.read_csv(doe_csv_filename)
        # We assume that if "GroundTruth_MIS" is NaN, the row has not been run.
        empty_rows = df[df["GroundTruth_MIS"].isna()]
        if empty_rows.empty:
            print("No more experiments in DOE.")
            return None, None, df
        row_idx = empty_rows.index[0]
        # Reserve this row by marking result columns with -1.
        for col in result_columns:
            df.at[row_idx, col] = -1
        df.to_csv(doe_csv_filename, index=False)
        return row_idx, df.loc[row_idx].to_dict(), df

def update_experiment_result(row_idx, result):
    """
    Reopens the DOE file with a lock, updates the row (with index row_idx) with result values,
    and saves the file.
    
    Parameters:
      row_idx: the integer index of the experiment row.
      result: a dictionary of result values (keys match result_columns).
    """
    with FileLock(lock_filename, timeout=10):
        df = pd.read_csv(doe_csv_filename)
        for col, val in result.items():
            df.at[row_idx, col] = val
        df.to_csv(doe_csv_filename, index=False)
        print(f"Updated experiment at row {row_idx} with results.")


# ===================== Evaluation Functions =====================

def create_bipartite_graph(n, coeff):
    """
    Create a bipartite graph with n nodes (n/2 in each partition).
    An edge between group A and group B is added with probability p = min(coeff*log(n)/n, 1).
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
    """Shuffle node ids while preserving node attributes."""
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
    Compute the minimum vertex cover for a bipartite graph G using the Hopcroft-Karp matching.
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
    """
    Compute the ground-truth maximal independent set (MIS) as nodes not in the minimum vertex cover.
    """
    mvc = compute_minimum_vertex_cover(G, groupA, groupB)
    mis = set(G.nodes()) - mvc
    return mis

def greedy_min_degree_mis(G):
    """Compute a greedy MIS using minimum degree deletion."""
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
    Simulate a noisy oracle for the MIS.
    For each vertex v:
      - If v is in I_star, mark 1 with probability (0.5 + epsilon), else 0.
      - If not in I_star, mark 0 with probability (0.5 + epsilon), else 1.
    Returns the set of vertices with output 1 and the full dictionary.
    """
    oracle = {}
    for v in G.nodes():
        if v in I_star:
            oracle[v] = 1 if random.random() < (0.5 + epsilon) else 0
        else:
            oracle[v] = 0 if random.random() < (0.5 + epsilon) else 1
    barI = {v for v, val in oracle.items() if val == 1}
    return barI, oracle


# ===================== Main Evaluation Workflow =====================

if __name__ == "__main__":
    while True:
        # Obtain a DOE experimental condition (with locking)
        row_index, experiment, doe_df = get_next_experiment()
        if experiment is None:
            print("All experiments have been processed. Exiting.")
            break

        print("Selected DOE experiment parameters:")
        print(experiment)
        
        # Convert parameters to appropriate types.
        n = int(experiment["Nodes"])
        coeff = float(experiment["Coefficient"])
        trial = int(experiment["Trial"])
        epsilon = float(experiment["Epsilon"])
        seed = int(experiment["Seed"])
        
        # Set the random seed.
        random.seed(seed)
        
        # Run the experiment.
        try:
            G, groupA, groupB = create_bipartite_graph(n, coeff)
        except Exception as e:
            print(f"Error creating graph: {e}")
            continue
        
        G, groupA, groupB = shuffle_node_ids(G)
        ground_truth_mis = compute_ground_truth_mis(G, groupA, groupB)
        greedy_mis = greedy_min_degree_mis(G)
        
        # Use the ground truth MIS as I_star for the noisy oracle simulation.
        I_star = ground_truth_mis
        barI, oracle = simulate_noisy_mis_oracle(G, I_star, epsilon)
        
        # Compute the noisy degree for each vertex.
        tilde_deg = {v: sum(1 for nb in G.neighbors(v) if nb in barI) for v in G.nodes()}
        
        # Compute a threshold s(v). Here we use Delta = average degree.
        avg_degree = sum(dict(G.degree()).values()) / float(G.number_of_nodes())
        Delta = avg_degree
        s = {v: (0.5 - epsilon) * G.degree(v) + 4 * math.sqrt(math.log(Delta)) * (0.5 - epsilon) * math.sqrt(Delta)
             for v in G.nodes()}
        
        S = {v for v in G.nodes() if tilde_deg[v] <= s[v]}
        H = G.subgraph(S)
        mis_noise = greedy_min_degree_mis(H)
        
        # Prepare results.
        result = {
            "GroundTruth_MIS": len(ground_truth_mis),
            "Greedy_MIS": len(greedy_mis),
            "Size_S": len(S),
            "Oracle_MIS": len(mis_noise),
            "Improvement": len(mis_noise) - len(greedy_mis)
        }
        
        print(f"Results for n={n}, coeff={coeff}, trial={trial}, epsilon={epsilon}:")
        print(result)
        
        # Update the DOE file with the experiment results.
        update_experiment_result(row_index, result)
        
        # Optionally, you can add a delay or any other handling before continuing.
