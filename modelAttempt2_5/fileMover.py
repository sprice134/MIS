# import os
# import shutil

# def move_graphs(source_dir, target_dir):
#     """
#     Moves all graph files from each node and removal percent subdirectory in source_dir
#     to the corresponding subdirectory in target_dir.
#     """
#     # Iterate over the "nodes_*" directories in the source directory.
#     for node_folder in os.listdir(source_dir):
#         source_node_path = os.path.join(source_dir, node_folder)
#         if os.path.isdir(source_node_path):
#             # Ensure the corresponding node directory exists in target_dir.
#             target_node_path = os.path.join(target_dir, node_folder)
#             os.makedirs(target_node_path, exist_ok=True)
            
#             # Iterate over the "removal_*percent" subdirectories.
#             for removal_folder in os.listdir(source_node_path):
#                 source_removal_path = os.path.join(source_node_path, removal_folder)
#                 if os.path.isdir(source_removal_path):
#                     # Ensure the corresponding removal directory exists in target_node_path.
#                     target_removal_path = os.path.join(target_node_path, removal_folder)
#                     os.makedirs(target_removal_path, exist_ok=True)
                    
#                     # Move each file from the source removal folder to the target removal folder.
#                     for filename in os.listdir(source_removal_path):
#                         source_file = os.path.join(source_removal_path, filename)
#                         target_file = os.path.join(target_removal_path, filename)
#                         shutil.move(source_file, target_file)
#                         print(f"Moved: {source_file} -> {target_file}")

# if __name__ == "__main__":
#     # Define the source and target directories.
#     source_dir = "generated_graphs_second_stage"
#     target_dir = "generated_graphs"
    
#     # Ensure that the target directory exists.
#     os.makedirs(target_dir, exist_ok=True)
    
#     move_graphs(source_dir, target_dir)
#     print("All graphs have been moved successfully.")


import os
import json

def append_json_files(source_dir, target_dir):
    """
    For each JSON file in source_dir, load its content (a list of results) and append that content
    to the like-named file in target_dir. If the target file doesn't exist, it will be created.
    """
    # Iterate over all JSON files in the source directory.
    for filename in os.listdir(source_dir):
        if not filename.endswith('.json'):
            continue  # Skip non-JSON files

        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)

        # Load JSON data from the source file.
        try:
            with open(source_file, 'r') as sf:
                source_data = json.load(sf)
        except Exception as e:
            print(f"Error reading {source_file}: {e}")
            continue

        # Ensure the source data is a list.
        if not isinstance(source_data, list):
            print(f"Expected a list in {source_file} but got {type(source_data).__name__}. Skipping.")
            continue

        # Load existing target data if the file exists, else start with an empty list.
        if os.path.exists(target_file):
            try:
                with open(target_file, 'r') as tf:
                    target_data = json.load(tf)
            except Exception as e:
                print(f"Error reading {target_file}: {e}. Starting with an empty list.")
                target_data = []
            if not isinstance(target_data, list):
                print(f"Expected a list in {target_file} but got {type(target_data).__name__}. Overwriting with an empty list.")
                target_data = []
        else:
            target_data = []

        # Append (i.e., extend) the target list with the source data.
        target_data.extend(source_data)

        # Write the combined data back to the target file.
        try:
            with open(target_file, 'w') as tf:
                json.dump(target_data, tf, indent=2)
            print(f"Appended data from '{source_file}' to '{target_file}'.")
        except Exception as e:
            print(f"Error writing to {target_file}: {e}")

def main():
    source_dir = "mis_results_grouped_v3_second_stage"
    target_dir = "mis_results_grouped_v3"

    if not os.path.isdir(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    if not os.path.isdir(target_dir):
        print(f"Target directory '{target_dir}' does not exist. Creating it.")
        os.makedirs(target_dir, exist_ok=True)

    append_json_files(source_dir, target_dir)
    print("All JSON files have been processed and appended.")

if __name__ == "__main__":
    main()
