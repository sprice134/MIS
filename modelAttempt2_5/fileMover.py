import os
import shutil

def move_graphs(source_dir, target_dir):
    """
    Moves all graph files from each node and removal percent subdirectory in source_dir
    to the corresponding subdirectory in target_dir.
    """
    # Iterate over the "nodes_*" directories in the source directory.
    for node_folder in os.listdir(source_dir):
        source_node_path = os.path.join(source_dir, node_folder)
        if os.path.isdir(source_node_path):
            # Ensure the corresponding node directory exists in target_dir.
            target_node_path = os.path.join(target_dir, node_folder)
            os.makedirs(target_node_path, exist_ok=True)
            
            # Iterate over the "removal_*percent" subdirectories.
            for removal_folder in os.listdir(source_node_path):
                source_removal_path = os.path.join(source_node_path, removal_folder)
                if os.path.isdir(source_removal_path):
                    # Ensure the corresponding removal directory exists in target_node_path.
                    target_removal_path = os.path.join(target_node_path, removal_folder)
                    os.makedirs(target_removal_path, exist_ok=True)
                    
                    # Move each file from the source removal folder to the target removal folder.
                    for filename in os.listdir(source_removal_path):
                        source_file = os.path.join(source_removal_path, filename)
                        target_file = os.path.join(target_removal_path, filename)
                        shutil.move(source_file, target_file)
                        print(f"Moved: {source_file} -> {target_file}")

if __name__ == "__main__":
    # Define the source and target directories.
    source_dir = "generated_graphs_second_stage"
    target_dir = "generated_graphs"
    
    # Ensure that the target directory exists.
    os.makedirs(target_dir, exist_ok=True)
    
    move_graphs(source_dir, target_dir)
    print("All graphs have been moved successfully.")
