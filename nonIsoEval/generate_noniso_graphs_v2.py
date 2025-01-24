#!/usr/bin/env python3

import itertools
import networkx as nx
import os
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

def write_edgelist_with_isolated_nodes(G, filename):
    """
    Write the edge list to a file, including isolated nodes as single-node lines.
    """
    with open(filename, 'w') as f:
        # Write all edges
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
        # Write isolated nodes
        isolated = list(nx.isolates(G))
        for node in isolated:
            f.write(f"{node}\n")

def worker(subsets, n):
    """
    Worker function to process a chunk of subsets.
    
    :param subsets: A list of edge subsets (each subset is a tuple of edges)
    :param n: Number of nodes in the graph
    :return: A local representative map {degseq: [Graph, ...]}
    """
    local_rep_map = {}
    all_nodes = list(range(n))
    
    for subset in subsets:
        # Build the graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(subset)

        # Compute degree sequence
        degseq = tuple(sorted(dict(G.degree()).values()))

        # Initialize list for this degree sequence if not present
        if degseq not in local_rep_map:
            local_rep_map[degseq] = [G]
        else:
            # Check isomorphism with existing representatives in local map
            is_new = True
            for rep in local_rep_map[degseq]:
                if nx.is_isomorphic(G, rep):
                    is_new = False
                    break
            if is_new:
                local_rep_map[degseq].append(G)
    
    return local_rep_map

def merge_rep_maps(global_map, local_map):
    """
    Merge a local representative map into the global map.
    
    :param global_map: The global representative map {degseq: [Graph, ...]}
    :param local_map: The local representative map to merge
    """
    for degseq, graphs in local_map.items():
        if degseq not in global_map:
            global_map[degseq] = graphs.copy()
        else:
            for G in graphs:
                is_new = True
                for rep in global_map[degseq]:
                    if nx.is_isomorphic(G, rep):
                        is_new = False
                        break
                if is_new:
                    global_map[degseq].append(G)

def chunkify(iterable, chunk_size):
    """
    Break an iterable into chunks of length 'chunk_size'.
    
    :param iterable: An iterable to chunkify
    :param chunk_size: Size of each chunk
    :return: Yields chunks as lists
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def generate_nonisomorphic_graphs_parallel(n=8, outdir="noniso_8_networkx_parallel", num_workers=None, chunk_size=1000):
    """
    Generate all non-isomorphic undirected graphs on n vertices using NetworkX.
    The process is parallelized to utilize multiple CPU cores.
    Each unique representative graph is written to an edgelist file.
    
    :param n: Number of nodes
    :param outdir: Output directory for edgelist files
    :param num_workers: Number of parallel worker processes (default: number of CPUs)
    :param chunk_size: Number of subsets per chunk for processing
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    all_nodes = list(range(n))
    all_edges = list(itertools.combinations(all_nodes, 2))
    num_edges = len(all_edges)
    total_subsets = 2 ** num_edges

    print(f"Generating non-isomorphic graphs for n={n}...")
    print(f"Total subsets to process: {total_subsets}")
    print(f"Using {num_workers or cpu_count()} workers and chunk size of {chunk_size}")

    # Create an iterator for all subsets
    subsets_iter = itertools.chain.from_iterable(itertools.combinations(all_edges, r) for r in range(num_edges + 1))
    
    # Create chunks of subsets
    chunks = chunkify(subsets_iter, chunk_size)

    # Initialize the global representative map
    global_rep_map = {}

    # Define the worker function with partial to fix 'n'
    worker_func = partial(worker, n=n)

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    # Create a multiprocessing pool
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for potentially better performance
        for local_map in tqdm(pool.imap_unordered(worker_func, chunks), 
                              total=total_subsets // chunk_size + 1, 
                              desc="Processing chunks"):
            merge_rep_maps(global_rep_map, local_map)

    # Collect all representative graphs from the global map
    representatives = []
    for graphs in global_rep_map.values():
        representatives.extend(graphs)

    print(f"Total non-isomorphic graphs found: {len(representatives)}")
    print(f"Writing graphs to '{outdir}'...")

    # Prepare arguments for writing graphs
    args_list = [ (G, idx, outdir) for idx, G in enumerate(representatives) ]

    # Define a global write function to avoid pickling issues
    def write_graph(args):
        G, idx, outdir = args
        filename = os.path.join(outdir, f"graph_{idx}.edgelist")
        write_edgelist_with_isolated_nodes(G, filename)

    # Write graphs in parallel
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(write_graph, args_list), 
                  total=len(args_list), 
                  desc="Writing graphs"))

    print(f"All non-isomorphic graphs have been written to '{outdir}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate all non-isomorphic undirected graphs for a given number of nodes with parallel processing."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=8,
        help="Number of nodes in the graphs (default: 8)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for edgelist files. If not specified, defaults to 'noniso_{n}_networkx_parallel'"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: number of CPUs)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of subsets per chunk for processing (default: 1000)"
    )
    args = parser.parse_args()

    n = args.nodes
    outdir = args.outdir if args.outdir else f"noniso_{n}_networkx_parallel"
    num_workers = args.workers
    chunk_size = args.chunk_size

    generate_nonisomorphic_graphs_parallel(n=n, outdir=outdir, num_workers=num_workers, chunk_size=chunk_size)

if __name__ == "__main__":
    main()

    # python generate_noniso_graphs_v2.py --nodes 8 --outdir noniso_8_networkx