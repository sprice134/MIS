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
  - Computes the MIS size using the persistent-noise algorithm (bravemen).
  - Writes a CSV row per graph with: num_nodes, removal_percent, graph_id, file_graph_id, true_mis, bravemen_mis.

In addition, the file defines a persistent-noise MIS algorithm (Algorithm 1) which uses an 
oracle with persistent noise (via the bravemenV1() function) to guide a greedy MIS selection.
"""

import os
import csv
import json
import argparse
import random
import math

import networkx as nx
from torch_geometric.utils import to_networkx
import sys
sys.path.append('../modelAttempt2_5')
from tools3 import MISGraphDataset  # Ensure this is accessible

#############################################
# Section 1: Functions for MIS Graph Extraction
#############################################

def gather_json_and_edgelist_paths(json_dir, node_counts, removal_percents):
    """
    For each combination of node count and removal percentage, build the expected JSON file
    path (from json_dir) and the corresponding edgelist directory path.
    Returns two lists of equal length.
    """
    json_paths = []
    edgelist_dirs = []
    
    # Assume edgelist directories are under: <parent_dir>/test_generated_graphs
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
    """
    Loads the JSON file and returns a dictionary mapping the edgelist file basename
    (e.g. "graph_100.edgelist") to its MIS info.
    """
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
    """
    Returns a sorted list of edgelist filenames from the given directory.
    """
    try:
        files = [f for f in os.listdir(edgelist_dir) if f.endswith(".edgelist")]
        return sorted(files)
    except Exception as e:
        print(f"Error listing edgelist files in {edgelist_dir}: {e}")
        return []


def extract_graph_id(file_key):
    """
    Given an edgelist file name like "graph_10.edgelist", extract and return the numeric graph ID.
    """
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
# Section 2: Persistent-Noise MIS Algorithm Functions
#############################################

def bravemenV1(mis_set, num_cells, epsilon):
    """
    Generate an oracle vector for a given MIS set with noisy inversion.

    For each cell in a graph:
      - Set it to 1 if its index is in mis_set, 0 otherwise.
      - Then, flip the cell's value with probability (0.5 - epsilon).
      
    This means a cell remains unchanged with probability (0.5 + epsilon).
    This aligns with the intended oracle behavior where for a vertex in the true MIS,
    the oracle outputs 1 with probability 0.5 + epsilon, and for a vertex not in the MIS,
    it outputs 0 with probability 0.5 + epsilon.
    
    Parameters:
        mis_set (set or list): Indices corresponding to cells that are in the MIS.
        num_cells (int): Total number of cells (nodes) in the graph.
        epsilon (float): The parameter controlling the noise (must be <= 0.5).
    
    Returns:
        oracleVector (list): A list of binary values (1 or 0) after applying the noise.
    """
    # Build initial binary vector: 1 for MIS cells, 0 for non-MIS cells.
    oracleVector = [1 if i in mis_set else 0 for i in range(num_cells)]
    
    # For each cell, flip its value with probability (0.5 - epsilon).
    for i in range(num_cells):
        if random.random() < (0.5 - epsilon):
            oracleVector[i] = 1 - oracleVector[i]
    
    return oracleVector


def greedy_mis_set(H):
    """
    A greedy algorithm to compute an independent set in graph H.
    This version returns the set of nodes selected.
    
    The algorithm repeatedly selects the node with minimum degree (with tie-breaking on the node ID)
    and removes that node and its neighbors.
    """
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
    """
    Implements Algorithm 1 for MIS in the persistent noise setting.
    
    Input:
      - G: A NetworkX graph (G = (V, E)).
      - mis_star: A set (or list) of vertex indices representing the maximum independent set I*.
                 (In practice, I* is unknown; here we assume it is provided for simulation.)
      - epsilon: A parameter (0 < epsilon < 0.5) controlling the persistence of the oracle's noise.
      
    Output:
      - I: An independent set of G computed as the greedy MIS on the induced subgraph G[S ∪ L],
           where S and L are defined according to Algorithm 1.
    
    Steps:
      1. Use bravemenV1() to generate a noisy oracle vector.
      2. For each vertex v, compute deg_gI*(v): the number of neighbors v with oracle value 1.
      3. Let L = {v ∈ V : deg(v) ≤ 36·ln(n)}.
      4. For v ∈ V \ L, compute s_v = (0.5 - epsilon)*deg(v) + 6√(ln n)*(0.5 - epsilon)*√(deg(v)).
         Let S = {v ∈ V \ L : deg_gI*(v) ≤ s_v}.
      5. Output the greedy MIS computed on the induced subgraph G[S ∪ L].
    """
    n = G.number_of_nodes()
    # Step 1: Generate the noisy oracle vector using bravemenV1.
    oracle_vector = bravemenV1(mis_star, n, epsilon)
    
    # Step 2: Compute deg_gI*(v) for each vertex v.
    deg_gI = {}
    for v in G.nodes():
        # Count how many neighbors are claimed to be in I* by the oracle.
        deg_gI[v] = sum(oracle_vector[w] for w in G.neighbors(v))
    
    # Step 3: Let L be the set of vertices with degree <= 36 * ln(n).
    L = {v for v in G.nodes() if G.degree(v) <= 36 * math.log(n)}
    
    # Step 4: For v in V \ L, compute s_v and let S be those with deg_gI(v) <= s_v.
    S = set()
    for v in G.nodes():
        if v not in L:
            d = G.degree(v)
            s_v = (0.5 - epsilon) * d + 6 * math.sqrt(math.log(n)) * (0.5 - epsilon) * math.sqrt(d)
            if deg_gI[v] <= s_v:
                S.add(v)
    
    # The induced subgraph on S ∪ L.
    induced_set = S.union(L)
    H = G.subgraph(induced_set).copy()
    
    # Step 5: Compute the greedy MIS on H.
    I = greedy_mis_set(H)
    
    return I

#############################################
# Section 3: Main Processing Function
#############################################

def main():
    parser = argparse.ArgumentParser(
        description="Extract true MIS sizes from MIS graphs and write results to CSV."
    )
    parser.add_argument("--node_counts", type=int, nargs="+", default=list(range(10, 55, 5)),
                        help="List of node counts to include, e.g. 10 15 20 ...")
    parser.add_argument("--removal_percents", type=int, nargs="+", default=list(range(15, 90, 5)),
                        help="List of removal percentages to include, e.g. 15 20 25 ...")
    parser.add_argument("--json_dir", type=str, default="/Users/sprice/Documents/GitHub/MIS/modelAttempt2_5/mis_results_grouped_v3",
                        help="Directory where JSON MIS info files are stored.")
    parser.add_argument("--csv_out", type=str, default="mis_basic_results.csv",
                        help="Output CSV filename.")
    parser.add_argument("--epsilon", type=float,
                        help="Probability that the oracle gives the correct bit (for persistent-noise MIS).")
    args = parser.parse_args()

    # Gather JSON file paths and corresponding edgelist directories.
    json_paths, edgelist_dirs = gather_json_and_edgelist_paths(
        json_dir=args.json_dir,
        node_counts=args.node_counts,
        removal_percents=args.removal_percents
    )
    
    if not json_paths or not edgelist_dirs:
        print("Error: No valid JSON files or edgelist directories found. Exiting.")
        return

    print(f"Found {len(json_paths)} (JSON, edgelist) pairs.")

    # CSV header with basic information plus the bravemen MIS size.
    csv_header = [
        "num_nodes", "removal_percent", "graph_id", "file_graph_id", "true_mis", "bravemen_mis"
    ]
    out_rows = []
    global_graph_id = 0

    # Process each (JSON, edgelist) pair.
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

        print(f"Processing edgelist dir '{edgelist_dir}' with removal_percent={percent_from_json}.")
        
        # Load graphs from this edgelist directory (and corresponding JSON).
        dataset = MISGraphDataset(
            json_paths=[json_path],
            edgelist_dirs=[edgelist_dir],
            label_type='prob'
        )
        print(f"  Loaded {len(dataset)} graphs from {edgelist_dir}.")

        if len(dataset) != len(edgelist_files):
            print(f"Warning: Number of graphs ({len(dataset)}) does not match number of edgelist files ({len(edgelist_files)}) in {edgelist_dir}.")

        for idx, data in enumerate(dataset):
            # Convert the PyG Data object to a NetworkX graph.
            G = to_networkx(data, to_undirected=True)
            # Ensure all nodes (including isolated ones) are present.
            G.add_nodes_from(range(data.num_nodes))
            num_nodes = data.num_nodes

            # Use the index to pick a file from edgelist_files.
            file_key = edgelist_files[idx] if idx < len(edgelist_files) else edgelist_files[-1]
            file_graph_id = extract_graph_id(file_key)
            true_mis = int(json_mapping.get(file_key, {}).get("MIS_SIZE", -1))
            
            # Use ground truth MIS from JSON instead of the greedy algorithm.
            json_entry = json_mapping.get(file_key, {})
            if "MIS_SETS" in json_entry and json_entry["MIS_SETS"]:
                # Use the first list in MIS_SETS as the ground truth independent set.
                mis_star = set(json_entry["MIS_SETS"][0])
            else:
                print(f"Warning: No ground truth MIS_SETS for file {file_key}. Using greedy MIS as fallback.")
                mis_star = greedy_mis_set(G)
            
            # Compute the independent set using the persistent-noise algorithm.
            bravemen_set = persistent_noise_mis(G, mis_star, args.epsilon)
            bravemen_size = len(bravemen_set)

            out_rows.append([
                num_nodes,
                percent_from_json,
                global_graph_id,
                file_graph_id,
                true_mis,
                bravemen_size
            ])
            print(f"  Graph {global_graph_id}: nodes={num_nodes}, TRUE_MIS={true_mis}, bravemen_mis={bravemen_size}")
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
    """
    An example function to demonstrate the persistent noise MIS algorithm on a random graph.
    """
    # Create a random graph.
    G = nx.erdos_renyi_graph(50, 0.1, seed=42)
    
    # For simulation, assume we know a maximum independent set (I*) using a greedy algorithm.
    mis_star = greedy_mis_set(G)
    print("Simulated I* (maximum independent set):", mis_star)
    
    epsilon = 0.8  # For example, 80% chance the oracle gives the correct bit.
    I = persistent_noise_mis(G, mis_star, epsilon)
    print("Computed independent set I using persistent-noise algorithm:", I)


#############################################
# Main Entry Point
#############################################

if __name__ == "__main__":
    # Uncomment one of the following lines depending on what you want to run:
    
    # Run the main MIS extraction and CSV writing process:
    main()
    
    # Or, to test the persistent-noise MIS algorithm on a sample graph, uncomment:
    # test_persistent_noise_mis()

    '''
    python attempt1.py \
    --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 \
    --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
    --json_dir /Users/sprice/Documents/GitHub/MIS/modelAttempt2_5/test_mis_results_grouped_v3 \
    --csv_out mis_basic_results_0.2.csv \
    --epsilon 0.2

    '''
