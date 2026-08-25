#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --job-name="pv_%j"
#SBATCH --output="logs/pvoros_%j.out"
#SBATCH --error="logs/pvoros_%j.err"

# Create the output directory if it doesn't exist
mkdir -p results_data

# Execute the python worker using the exported environment variables
python3 pvoros_w1w2.py --idx_i $i --idx_j $j --grid_size $GRID_SIZE