#!/usr/bin/env python3
"""
Simplified MIS Processing and Persistent-Noise MIS Algorithm

This script processes MIS graphs on a per-(JSON, edgelist) pair basis.
For each JSON file (named like:
    nodes_{num_nodes}_removal_{removal_percent}percent.json
and containing a list of graph dictionaries with MIS information) and its corresponding
edgelist directory, the script:
  - Loads the MIS JSON entries and builds a mapping from edgelist basename → MIS info.
  - Lists all edgelist files (sorted) from the edgelist directory.
  - Loads all graphs from the edgelist directory (using MISGraphDataset).
  - For each graph, uses the index in the mini-dataset to pick the corresponding edgelist filename.
  - Extracts the numeric graph ID from the edgelist filename (e.g. "graph_10.edgelist" → 10).
  - Extracts the true MIS size from the JSON entry.
  - Computes the MIS size using two methods:
      a) Greedy MIS on the full graph.
      b) The persistent-noise algorithm (Algorithm 3) that computes MIS on the induced subgraph G[S ∪ L],
         where:
           - Δ = maximum degree of G.
           - L = { v ∈ V : deg(v) ≤ Δ/2 }.
           - For vertices v ∉ L, a threshold s_v = (0.5 - ε)*deg(v) + 6*√(ln(Δ))*(0.5 - ε)*√(deg(v))
           - S_V = { v ∈ V∖L : (noisy degree from the oracle) ≤ s_v }.
  - Also computes the sizes of L, S_V, and (V∖L)∖S_V.
  - Writes a CSV row per graph with:
      num_nodes, removal_percent, graph_id, file_graph_id, true_mis,
      greedy_mis (full graph), algorithm3_mis (persistent-noise MIS),
      |L|, |S_V|, and |(V∖L)∖S_V|.
"""

import os
import csv
import json
import argparse
import random
import math
import re
import networkx as nx
import sys
sys.path.append('../modelAttempt2_5')
from tools3 import MISGraphDataset  # Ensure this is accessible
from torch_geometric.utils import to_networkx

#############################################
# Section 1: Functions for MIS Graph Extraction
#############################################

def gather_json_and_edgelist_paths(json_dir, node_counts, removal_percents):
    json_paths = []
    edgelist_dirs = []
    parent_dir = os.path.dirname(json_dir)
    edgelist_base = os.path.join(parent_dir, "test_generated_graphs")
    
    for n in node_counts:
        for percent in removal_percents:
            json_filename = f"nodes_{n}_removal_{percent}percent.json"
            json_path = os.path.join(json_dir, json_filename)
            if not os.path.exists(json_path):
                print(f"Warning: JSON file '{json_path}' does not exist. Skipping.")
                continue
            json_paths.append(json_path)
            
            edgelist_dir = os.path.join(edgelist_base, f"nodes_{n}", f"removal_{percent}percent")
            if not os.path.exists(edgelist_dir):
                print(f"Warning: Edgelist directory '{edgelist_dir}' does not exist. Skipping.")
                continue
            edgelist_dirs.append(edgelist_dir)
    
    return json_paths, edgelist_dirs


def build_json_mapping(json_path):
    mapping = {}
    try:
        with open(json_path, "r") as f:
            data_list = json.load(f)
        for entry in data_list:
            key = os.path.basename(entry.get("file_path", ""))
            if key:
                mapping[key] = entry
            else:
                print(f"Warning: An entry in {json_path} is missing 'file_path'.")
    except Exception as e:
        print(f"Error reading JSON from {json_path}: {e}")
    return mapping


def list_edgelist_files(edgelist_dir):
    try:
        files = [f for f in os.listdir(edgelist_dir) if f.endswith(".edgelist")]
        return sorted(files)
    except Exception as e:
        print(f"Error listing edgelist files in {edgelist_dir}: {e}")
        return []


def extract_graph_id(file_key):
    try:
        parts = file_key.split("_")
        if len(parts) >= 2:
            id_part = parts[1]
            id_str = id_part.split(".")[0]
            return int(id_str)
        else:
            print(f"Warning: Unexpected file key format: {file_key}")
            return -1
    except Exception as e:
        print(f"Error extracting graph id from {file_key}: {e}")
        return -1

#############################################
# Section 2: MIS Algorithm Functions
#############################################

def bravemenV1(mis_set, num_cells, epsilon):
    oracleVector = [1 if i in mis_set else 0 for i in range(num_cells)]
    for i in range(num_cells):
        if random.random() < (0.5 - epsilon):
            oracleVector[i] = 1 - oracleVector[i]
    return oracleVector

def greedy_mis_set(H):
    independent_set = set()
    H_copy = H.copy()
    while H_copy.number_of_nodes() > 0:
        node = min(H_copy.nodes(), key=lambda n: (H_copy.degree(n), n))
        independent_set.add(node)
        neighbors = list(H_copy.neighbors(node))
        H_copy.remove_node(node)
        H_copy.remove_nodes_from(neighbors)
    return independent_set

def persistent_noise_mis(G, mis_star, epsilon):
    # Compute oracle vector.
    n = G.number_of_nodes()
    oracle_vector = bravemenV1(mis_star, n, epsilon)
    deg_gI = {v: sum(oracle_vector[w] for w in G.neighbors(v)) for v in G.nodes()}
    max_degree = max(dict(G.degree()).values())
    # Compute L = { v in V : deg(v) <= Δ/2 }.
    L = {v for v in G.nodes() if G.degree(v) <= max_degree/2}
    # Compute S_V for vertices not in L.
    S = set()
    for v in G.nodes():
        if v not in L:
            d = G.degree(v)
            s_v = (0.5 - epsilon) * d + 6 * math.sqrt(math.log(max_degree)) * (0.5 - epsilon) * math.sqrt(d)
            if deg_gI[v] <= s_v:
                S.add(v)
    induced_set = S.union(L)
    I = greedy_mis_set(G.subgraph(induced_set))
    return I, L, S  # Return the computed independent set and the sets L and S.

#############################################
# Section 3: Main Processing Function
#############################################

def main():
    parser = argparse.ArgumentParser(
        description="Extract MIS sizes from multiple graphs and write results to CSV."
    )
    parser.add_argument("--node_counts", type=int, nargs="+", default=list(range(10, 55, 5)),
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", default=list(range(15, 90, 5)),
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--json_dir", type=str, default="/path/to/mis_results_grouped_v3",
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--csv_out", type=str, default="mis_basic_results.csv",
                        help="Output CSV filename.")
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Epsilon value for the persistent-noise MIS oracle (0 < epsilon < 0.5).")
    args = parser.parse_args()

    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(
        json_dir=args.json_dir,
        node_counts=args.node_counts,
        removal_percents=args.removal_percents
    )
    
    if not json_paths or not edgelist_dirs:
        print("Error: No valid JSON files or edgelist directories found. Exiting.")
        return

    print(f"Found {len(json_paths)} (JSON, edgelist) pairs.")

    # CSV header.
    csv_header = [
        "num_nodes", "removal_percent", "graph_id", "file_graph_id", "true_mis",
        "greedy_mis", "bravemen_mis", "L_len", "S_len", "V_minus_L_minus_S_len"
    ]
    out_rows = []
    global_graph_id = 0

    for json_path, edgelist_dir in zip(json_paths, edgelist_dirs):
        json_mapping = build_json_mapping(json_path)
        sample_entry = next(iter(json_mapping.values()), {})
        percent_from_json = sample_entry.get("percent_removed", None)
        if percent_from_json is None:
            print(f"Warning: No percent_removed found in {json_path}; using value from filename.")
            percent_from_json = int(os.path.basename(json_path).split("removal_")[1].split("percent")[0])
        
        edgelist_files = list_edgelist_files(edgelist_dir)
        if not edgelist_files:
            print(f"Warning: No edgelist files found in {edgelist_dir}; skipping this pair.")
            continue

        print(f"Processing edgelist directory '{edgelist_dir}' with removal_percent={percent_from_json}.")
        
        dataset = MISGraphDataset(
            json_paths=[json_path],
            edgelist_dirs=[edgelist_dir],
            label_type='prob'
        )
        print(f"  Loaded {len(dataset)} graphs from {edgelist_dir}.")

        if len(dataset) != len(edgelist_files):
            print(f"Warning: Number of graphs ({len(dataset)}) does not match number of edgelist files ({len(edgelist_files)}) in {edgelist_dir}.")

        for idx, data in enumerate(dataset):
            G = to_networkx(data, to_undirected=True)
            G.add_nodes_from(range(data.num_nodes))
            num_nodes = data.num_nodes

            file_key = edgelist_files[idx] if idx < len(edgelist_files) else edgelist_files[-1]
            file_graph_id = extract_graph_id(file_key)
            true_mis = int(json_mapping.get(file_key, {}).get("MIS_SIZE", -1))
            
            json_entry = json_mapping.get(file_key, {})
            if "MIS_SETS" in json_entry and json_entry["MIS_SETS"]:
                mis_star = set(json_entry["MIS_SETS"][0])
            else:
                print(f"Warning: No ground truth MIS_SETS for file {file_key}. Using greedy MIS as fallback.")
                mis_star = greedy_mis_set(G)
            
            # Compute greedy MIS on the full graph.
            greedy_full = greedy_mis_set(G)
            greedy_full_len = len(greedy_full)
            
            # Compute persistent-noise MIS (Algorithm 3) along with L and S.
            bravemen_set, L, S = persistent_noise_mis(G, mis_star, args.epsilon)
            bravemen_size = len(bravemen_set)
            L_len = len(L)
            S_len = len(S)
            # V \ L:
            V_minus_L = set(G.nodes()) - L
            # (V \ L) - S:
            V_minus_L_minus_S = V_minus_L - S
            remainder_len = len(V_minus_L_minus_S)
            
            out_rows.append([
                num_nodes,
                percent_from_json,
                global_graph_id,
                file_graph_id,
                true_mis,
                greedy_full_len,
                bravemen_size,
                L_len,
                S_len,
                remainder_len
            ])
            print(f"  Graph {global_graph_id}: nodes={num_nodes}, TRUE_MIS={true_mis}, greedy_mis={greedy_full_len}, bravemen_mis={bravemen_size}, |L|={L_len}, |S|={S_len}, |(V\\L)-S|={remainder_len}")
            global_graph_id += 1

    with open(args.csv_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        writer.writerows(out_rows)

    print(f"\nResults written to {args.csv_out}")


#############################################
# Section 4: Example Usage of Persistent-Noise MIS Algorithm
#############################################

def test_persistent_noise_mis():
    import networkx as nx
    G = nx.erdos_renyi_graph(50, 0.1, seed=42)
    mis_star = greedy_mis_set(G)
    print("Simulated I* (maximum independent set):", mis_star)
    epsilon = 0.2  # Example epsilon.
    I, L, S = persistent_noise_mis(G, mis_star, epsilon)
    print("Computed independent set I using persistent-noise algorithm:", I)
    print("L =", L)
    print("S =", S)

#############################################
# Main Entry Point
#############################################

if __name__ == "__main__":
    main()
    # To test on a sample graph, uncomment:
    # test_persistent_noise_mis()
    '''
    Example command line usage:
    python attempt3.py \
      --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 \
      --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
      --json_dir ../modelAttempt2_5/test_mis_results_grouped_v3 \
      --csv_out v3_mis_basic_results.csv \
      --epsilon 0.1
    '''
