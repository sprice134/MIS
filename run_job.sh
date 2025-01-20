#!/bin/bash
#SBATCH --job-name=job1137    # Job name
#SBATCH --cpus-per-task=64               # Number of CPU cores
#SBATCH --gres=gpu:0                    # Number of GPUs
#SBATCH --mem=64000MB                   # Memory in MB
#SBATCH --time=4:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=short               # Partition name

# Print some job information
echo "Running job on $SLURM_JOB_NODELIST"
echo "Requested resources:"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - GPUs: $SLURM_GPUS"
echo "  - Memory: $SLURM_MEM_PER_NODE"

# Activate the Python virtual environment
source /home/sprice/MIS/nodeCreator/bin/activate


# python compute_mis_commandLine_v2.py --node_counts 15 20 25 30 35 40 45 50 55 
# python compute_mis_commandLine_v2.py --node_counts 60 65 70 75 
# python compute_mis_commandLine_v2.py --node_counts 80 85 90 95
# python compute_mis_commandLine_v2.py --node_counts 100
# python compute_mis_commandLine_v2.py --node_counts 105
# python compute_mis_commandLine_v2.py --node_counts 110
# python compute_mis_commandLine_v2.py --node_counts 115
# python compute_mis_commandLine_v2.py --node_counts 120
python compute_mis_commandLine_v2.py --node_counts 90
