#!/bin/bash
#SBATCH --job-name=bipartite2                # Job name
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --gres=gpu:0                    # Number of GPUs
#SBATCH --mem=12000MB                   # Memory in MB
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=short               # Partition name

# Print some job information
echo "Running job on $SLURM_JOB_NODELIST"
echo "Requested resources:"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - GPUs: $SLURM_GPUS"
echo "  - Memory: $SLURM_MEM_PER_NODE"

# Activate the Python virtual environment
source /home/sprice/MIS/nodeCreator/bin/activate
python /home/sprice/MIS/_reichman/bipartite2/eval2.py
# python /home/sprice/MIS/_reichman/bipartite/oracleBipartiteExperimentCLtemp2.py 6500