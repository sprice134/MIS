#!/usr/bin/env python3

import itertools
import networkx as nx
import os
from tqdm import tqdm
import argparse

def write_edgelist_with_isolated_nodes(G, filename):
    """
    Write the edge list to a file, including isolated nodes as single-node lines.
    
    :param G: NetworkX graph
    :param filename: Output file path
    """
    with open(filename, 'w') as f:
        # Write all edges
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
        # Write isolated nodes
        isolated = list(nx.isolates(G))
        for node in isolated:
            f.write(f"{node}\n")

def generate_nonisomorphic_graphs_networkx(n=5, outdir="noniso_5_networkx"):
    """
    Generate all non-isomorphic undirected graphs on n vertices using NetworkX.
    The approach:
      1) Enumerate all subsets of edges (2^(n*(n-1)/2) total).
      2) Build a graph for each subset.
      3) Use sorted degree sequence to group potential isomorphs.
      4) Use NX is_isomorphic to confirm isomorphism, only store unique reps.
    Writes each representative graph to its own file in 'outdir'.
    
    :param n: Number of nodes
    :param outdir: Output directory for edgelist files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Store representative graphs keyed by their sorted degree sequence
    rep_map = {}  # dict: degree_seq (tuple) -> list of representative Graphs

    # List all possible edges among n vertices
    all_nodes = range(n)
    all_edges = list(itertools.combinations(all_nodes, 2))
    num_edges = len(all_edges)

    # Total number of subsets of edges
    num_subsets = 2 ** num_edges

    # Use tqdm to display progress over all subsets of edges
    with tqdm(total=num_subsets, desc="Generating subsets") as pbar:
        # Iterate through all subsets of edges by subset size
        for subset_size in range(num_edges + 1):
            for subset in itertools.combinations(all_edges, subset_size):
                # Build the graph
                G = nx.Graph()
                G.add_nodes_from(all_nodes)
                G.add_edges_from(subset)

                # Compute a simple "fingerprint": the sorted degree sequence
                degseq = tuple(sorted(dict(G.degree()).values()))

                # Check if there's an existing rep in rep_map[degseq]
                is_new = True
                if degseq in rep_map:
                    for rep in rep_map[degseq]:
                        if nx.is_isomorphic(G, rep):
                            is_new = False
                            break
                else:
                    rep_map[degseq] = []

                # If this graph is not isomorphic to an existing rep, store it
                if is_new:
                    rep_map[degseq].append(G)

                # Update the tqdm progress bar
                pbar.update(1)

    # Collect all representative graphs from rep_map
    representatives = []
    for degseq, graphs in rep_map.items():
        representatives.extend(graphs)

    # Write each representative to a file, including isolated nodes
    for idx, graph in enumerate(tqdm(representatives, desc="Writing graphs")):
        filename = os.path.join(outdir, f"graph_{idx}.edgelist")
        write_edgelist_with_isolated_nodes(graph, filename)

    print(f"Found {len(representatives)} non-isomorphic graphs in '{outdir}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate all non-isomorphic undirected graphs for a given number of nodes."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="Number of nodes in the graphs (default: 5)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for edgelist files. If not specified, defaults to 'noniso_{n}_networkx'"
    )
    args = parser.parse_args()

    n = args.nodes
    outdir = args.outdir if args.outdir else f"noniso_{n}_networkx"

    generate_nonisomorphic_graphs_networkx(n=n, outdir=outdir)

if __name__ == "__main__":
    main()
