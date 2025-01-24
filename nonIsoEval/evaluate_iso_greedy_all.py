#!/usr/bin/env python3

import os
import argparse
import json
import networkx as nx
from itertools import permutations, islice
from collections import Counter
from tqdm import tqdm
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import csv

def load_brute_force_mis(json_file):
    """
    Loads the brute-force MIS JSON file and returns a dictionary keyed by normalized file_path.
    Each value is the MIS size.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    results_dict = {}
    for entry in data:
        file_path = entry.get("file_path")
        mis_size = entry.get("mis_size")
        if file_path and mis_size is not None:
            # Normalize file_path to match the graph_dir paths
            normalized_path = os.path.normpath(file_path)
            results_dict[normalized_path] = mis_size
    return results_dict

def load_greedy_mis(greedy_json_file):
    """
    Loads the greedy MIS JSON file and returns a dictionary keyed by normalized file_path.
    Each value is the MIS size.
    """
    with open(greedy_json_file, "r") as f:
        data = json.load(f)
    
    results_dict = {}
    for entry in data:
        file_path = entry.get("file_path")
        mis_size = entry.get("greedy_mis_size")
        if file_path and mis_size is not None:
            # Normalize file_path to match the graph_dir paths
            normalized_path = os.path.normpath(file_path)
            results_dict[normalized_path] = mis_size
    return results_dict

def load_graph(file_path, n):
    """
    Loads a graph from an edge list file and ensures all nodes are included.
    """
    G = nx.read_edgelist(file_path, nodetype=int)
    G.add_nodes_from(range(n))
    return G

def greedy_mis(G):
    """
    Greedy algorithm to find an independent set by repeatedly selecting
    the node with the fewest neighbors, breaking ties by the smallest label.
    """
    independent_set = []
    H = G.copy()
    while H.nodes():
        # Select the node with minimum (degree, label)
        node = min(H.nodes(), key=lambda x: (H.degree(x), x))
        independent_set.append(node)
        # Remove the chosen node and its neighbors from the graph
        neighbors = list(H.neighbors(node))
        H.remove_node(node)
        H.remove_nodes_from(neighbors)
    return independent_set

def process_permutation(args):
    """
    Processes a single permutation: relabels the graph, runs greedy MIS,
    and checks if the MIS size matches the true MIS size.
    
    Args:
        args: Tuple containing (permutation tuple, graph_edges, n, true_mis_size)
    
    Returns:
        1 if matched, 0 otherwise
    """
    perm, graph_edges, n, true_mis_size = args
    # Create mapping from old to new labels
    mapping = {old: new for old, new in zip(range(n), perm)}
    # Relabel the graph
    relabeled_edges = [(mapping[edge[0]], mapping[edge[1]]) for edge in graph_edges]
    G_iso = nx.Graph()
    G_iso.add_nodes_from(range(n))
    G_iso.add_edges_from(relabeled_edges)
    # Run greedy MIS
    mis_set = greedy_mis(G_iso)
    mis_size = len(mis_set)
    # Compare
    return 1 if mis_size == true_mis_size else 0

def chunked_permutations(n, chunk_size=10000):
    """
    Generator that yields chunks of permutations.
    
    Args:
        n: Number of nodes
        chunk_size: Number of permutations per chunk
    
    Yields:
        List of permutation tuples
    """
    perm_gen = permutations(range(n))
    while True:
        chunk = list(islice(perm_gen, chunk_size))
        if not chunk:
            break
        yield chunk

def evaluate_graph(args):
    """
    Evaluates a single graph: generates all permutations, runs greedy MIS,
    and counts matches/mismatches.
    
    Args:
        args: Tuple containing (graph_id, graph_path, true_mis_size, n)
    
    Returns:
        Tuple containing (graph_id, matched_count, mismatched_count, classification)
    """
    graph_id, graph_path, true_mis_size, n = args
    G = load_graph(graph_path, n)
    graph_edges = list(G.edges())
    
    total_perms = math.factorial(n)
    matched_count = 0
    mismatched_count = 0
    
    # Process permutations in chunks without creating a new Pool
    for perm_chunk in tqdm(chunked_permutations(n, chunk_size=10000), 
                           total=math.factorial(n) // 10000 + 1, 
                           desc=f"Processing {graph_id} Permutations"):
        # Prepare arguments for the current chunk
        args_list = [ (perm, graph_edges, n, true_mis_size) for perm in perm_chunk ]
        # Map the process_permutation function over the chunk
        results = list(map(process_permutation, args_list))
        # Update counts
        matched_count += sum(results)
        mismatched_count += len(results) - sum(results)
    
    # Assign classification
    if matched_count == total_perms:
        classification = 1
    elif matched_count == 0:
        classification = 3
    else:
        classification = 2
    
    return (graph_id, matched_count, mismatched_count, classification)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Greedy MIS against Brute-Force MIS for all graphs in a directory."
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Directory containing .edgelist graph files."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        required=True,
        help="Number of nodes in the graphs."
    )
    parser.add_argument(
        "--mis_json",
        type=str,
        required=True,
        help="Path to the JSON file with brute-force MIS results."
    )
    parser.add_argument(
        "--greedy_mis_json",
        type=str,
        required=True,
        help="Path to the JSON file with greedy MIS results."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file."
    )
    args = parser.parse_args()
    
    graph_dir = args.graph_dir
    n = args.nodes
    mis_json = args.mis_json
    greedy_mis_json = args.greedy_mis_json
    output_csv = args.output_csv
    
    # Validate graph directory
    if not os.path.isdir(graph_dir):
        print(f"Error: The directory '{graph_dir}' does not exist.")
        return
    
    # Load brute-force MIS results
    print("Loading brute-force MIS results...")
    brute_force_results = load_brute_force_mis(mis_json)
    
    # Load greedy MIS results
    print("Loading greedy MIS results...")
    greedy_results = load_greedy_mis(greedy_mis_json)
    
    # Prepare list of graphs to process
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith(".edgelist")]
    if not graph_files:
        print(f"No .edgelist files found in directory '{graph_dir}'.")
        return
    
    # Prepare arguments for each graph
    graphs_to_evaluate = []
    for graph_file in graph_files:
        graph_id = os.path.splitext(graph_file)[0]  # Filename without extension
        graph_path = os.path.normpath(os.path.join(graph_dir, graph_file))
        # Retrieve true MIS size
        true_mis_size = brute_force_results.get(graph_path)
        if true_mis_size is None:
            print(f"Warning: No MIS size found for '{graph_path}' in the brute-force JSON file. Skipping.")
            continue
        # Retrieve greedy MIS size
        greedy_mis_size = greedy_results.get(graph_path)
        if greedy_mis_size is None:
            print(f"Warning: No Greedy MIS size found for '{graph_path}' in the greedy MIS JSON file. Skipping.")
            continue
        # Note: Greedy MIS per permutation is not needed here since we are recalculating it
        graphs_to_evaluate.append( (graph_id, graph_path, true_mis_size, n) )
    
    if not graphs_to_evaluate:
        print("No graphs to evaluate. Exiting.")
        return
    
    # Use multiprocessing to evaluate graphs in parallel
    num_workers = cpu_count()
    print(f"\nEvaluating {len(graphs_to_evaluate)} graphs using {num_workers} workers...\n")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(evaluate_graph, graphs_to_evaluate), 
                            total=len(graphs_to_evaluate), 
                            desc="Evaluating Graphs"))
    
    # Write results to CSV
    print(f"\nWriting results to '{output_csv}'...")
    with open(output_csv, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(["Graph_ID", "Matched_Count", "Mismatched_Count", "Classification"])
        # Write data
        for row in results:
            csvwriter.writerow(row)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()

    # python evaluate_iso_greedy_all.py --graph_dir noniso_7_networkx --nodes 7 --mis_json mis_results_7.json --greedy_mis_json greedy_mis_results_7.json --output_csv evaluation_results.csv

