#!/bin/bash
#SBATCH --job-name=bipartite_eval              # Job name
#SBATCH --cpus-per-task=32              # Number of CPU cores
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --mem=64000MB                   # Memory in MB
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


# python compute_mis_commandLine_v2.py --node_counts 15 20 25 30 35 40 45 50 55 
# python compute_mis_commandLine_v2.py --node_counts 60 65 70 75 
# python compute_mis_commandLine_v2.py --node_counts 80 85 90 95
# python compute_mis_commandLine_v2.py --node_counts 100
# python compute_mis_commandLine_v2.py --node_counts 105
# python compute_mis_commandLine_v2.py --node_counts 110
# python compute_mis_commandLine_v2.py --node_counts 115
# python compute_mis_commandLine_v2.py --node_counts 120
cd modelAttempt2_5

# python modelTrain_prob.py \
#         --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 75\
#         --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
#         --output_dir mis_results_grouped_v3 \
#         --batch_size 64\
#         --hidden_channels 125 \
#         --num_layers 64 \
#         --learning_rate 0.005 \
#         --epochs 1000 \
#         --patience 20 \
#         --model_save_path best_model_prob_64_125_64_0.005_v3.pth

# python modelTrain_prob.py \
#         --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90\
#         --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
#         --output_dir mis_results_grouped_v3 \
#         --batch_size 32\
#         --hidden_channels 176 \
#         --num_layers 28 \
#         --learning_rate 0.0010221252698628714 \
#         --epochs 1000 \
#         --patience 35 \
#         --model_save_path best_model_prob_32_176_28_0.001_v5.pth

# python modelTrain_prob.py \
#         --node_counts 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90\
#         --removal_percents 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 \
#         --output_dir mis_results_grouped_v3 \
#         --batch_size 32\
#         --hidden_channels 176 \
#         --num_layers 28 \
#         --learning_rate 0.0010221252698628714 \
#         --epochs 1000 \
#         --patience 35 \
#         --model_save_path best_model_prob_32_176_28_0.001_v6.pth

python greedyEvalV3_bipartite.py \
      --node_counts 100 200 300 400 500 600 700 800 900 1000 \
      --bipartite_numbers 5 8 11 14 17 20 23 26 29 32 35 38 41 44 47 \
      --json_dir /home/sprice/MIS/bipartite/mis_results_grouped \
      --edgelist_base /home/sprice/MIS/bipartite/generated_graphs_bipartite \
      --csv_out mis_greedy_results_bipartite.csv \
      --model_path best_model_prob_32_176_28_0.001_v6.pth \
      --hidden_channels 176 \
      --num_layers 28

# python misEvaluator.py --node_counts 55 --base_dir generated_graphs --output_dir mis_results_grouped_v3
# python misEvaluator.py --node_counts 60 --base_dir generated_graphs --output_dir mis_results_grouped_v3
# python misEvaluator.py --node_counts 65 --base_dir test_generated_graphs --output_dir test_mis_results_grouped_v3

# python dataCreator.py
# python misEvaluator_optimized.py --node_counts 80 --base_dir test_generated_graphs --output_dir test_mis_results_grouped_v3 --removal_percents 85
# python misEvaluator_optimized.py --node_counts 95 --base_dir test_generated_graphs --output_dir test_mis_results_grouped_v3
# python misEvaluator_optimized.py --node_counts 90 --base_dir test_generated_graphs --output_dir test_mis_results_grouped_v3 --removal_percents 85 --temp_backup
# python optuna_mis_train.py --n_trials 25 --output_dir mis_results_grouped_v3
# python misEvaluator_optimized.py --node_counts 90 --base_dir generated_graphs --output_dir mis_results_grouped_v3 --removal_percents 80 85

# python compute_greedy_mis_commandline.py --node_counts 100 105 110 115 120 125 130 135 140 145 150
