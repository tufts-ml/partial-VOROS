## Run: bash w1w2_sweep.sh 

#!/bin/bash

GRID_SIZE=30

mkdir -p logs
mkdir -p results_data

for i in $(seq 0 $((GRID_SIZE - 1))); do
    for j in $(seq 0 $((GRID_SIZE - 1))); do
        
        # Directly sbatch the .slurm file while setting unique names and exporting variables
        sbatch --job-name="pv_${i}_${j}" \
               --output="logs/pvoros_${i}_${j}.out" \
               --error="logs/pvoros_${i}_${j}.err" \
               --export=ALL,IDX_I="$i",IDX_J="$j",GRID_SIZE="$GRID_SIZE" \
               run_pvoros.slurm
    done
done