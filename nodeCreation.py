import os
import random
import networkx as nx

# Define parameters
node_counts = list(range(15, 155, 5))  # 15, 20, ..., 100
removal_percents = list(range(15, 90, 5))  # 15%, 20%, ..., 85%
iterations = 50

# Base directory to store generated graphs
base_dir = "generated_graphs"

# Create the base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

for n in node_counts:
    # Create a directory for each node count
    node_dir = os.path.join(base_dir, f"nodes_{n}")
    os.makedirs(node_dir, exist_ok=True)
    
    # Generate a complete graph with n nodes once for reuse 
    complete_graph = nx.complete_graph(n)
    all_edges = list(complete_graph.edges())
    total_edges = len(all_edges)
    
    for percent in removal_percents:
        # Directory for this percentage of removal
        percent_dir = os.path.join(node_dir, f"removal_{percent}percent")
        os.makedirs(percent_dir, exist_ok=True)
        
        # Determine how many edges to remove
        num_remove = int(total_edges * (percent / 100.0))
        
        for iteration in range(1, iterations + 1):
            # Copy the complete graph for this iteration
            G = complete_graph.copy()
            
            # Randomly sample edges to remove
            edges_to_remove = random.sample(all_edges, num_remove)
            G.remove_edges_from(edges_to_remove)
            
            # Optionally: Check connectivity or other properties if needed
            
            # Save the graph to a file
            filename = os.path.join(percent_dir, f"graph_{iteration}.edgelist")
            nx.write_edgelist(G, filename, data=False)
            
            print(f"Saved graph: Nodes={n}, Removal={percent}%, Iteration={iteration}")
