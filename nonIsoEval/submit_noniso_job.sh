#!/bin/bash

# Define the number of nodes as an environment variable.
# This variable will be passed to SBATCH directives and the script itself.
# When submitting the job, you'll set this variable using the --export option.

#SBATCH --job-name=noniso_${NODES}     # Job name includes the number of nodes
#SBATCH --cpus-per-task=64             # Number of CPU cores
#SBATCH --gres=gpu:0                   # Number of GPUs
#SBATCH --mem=64000MB                  # Memory in MB
#SBATCH --time=48:00:00                # Time limit (HH:MM:SS)
#SBATCH --partition=long               # Partition name

# ======================================================================
# Instructions:
# To submit this script with a specific number of nodes, use the following command:
#
#     sbatch --export=NODES=7 submit_noniso_job.sh
#
# Replace '7' with your desired number of nodes.
# ======================================================================

# Print some job information
echo "Running job on $SLURM_JOB_NODELIST"
echo "Requested resources:"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - GPUs: $SLURM_GPUS"
echo "  - Memory: $SLURM_MEM_PER_NODE"
echo "  - Number of Nodes: $NODES"

# Activate the Python virtual environment
source /home/sprice/MIS/nodeCreator/bin/activate

# Execute the Python scripts with the specified number of nodes
python generate_noniso_graphs.py --nodes "$NODES" --outdir "noniso_${NODES}_networkx"
python calculate_mis.py --nodes "$NODES" --input_dir "noniso_${NODES}_networkx" --output_file "mis_results_${NODES}.json"
python calculate_greedy_mis.py --nodes "$NODES" --input_dir "noniso_${NODES}_networkx" --output_file "greedy_mis_results_${NODES}.json"
python analysis.py --nodes "$NODES" --gt "mis_results_${NODES}.json" --greedy "greedy_mis_results_${NODES}.json"

# Optional: Deactivate the virtual environment after the job is done
deactivate
