#!/bin/bash
#SBATCH --job-name=soft_pvoros
#SBATCH --output=logs/soft_pv_%j.out
#SBATCH --error=logs/soft_pv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00


# Create the output directory if it doesn't exist
mkdir -p results_data_soft_pv

# Execute the python worker using the exported environment variables
echo "i=$i, j=$j"
python3 soft_pv_w1w2.py --idx_i $i --idx_j $j --grid_size $GRID_SIZE