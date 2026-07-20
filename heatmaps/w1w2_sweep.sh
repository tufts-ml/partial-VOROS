## Run: bash w1w2_sweep.sh (sigmoid)

#!/bin/bash

GRID_SIZE=30

mkdir -p logs
mkdir -p results_data
mkdir -p results_data_soft_pv

if [[ -z $1 ]]; then
    SIGMOID='none'
else
    SIGMOID=$1
fi

for i in $(seq 0 $((GRID_SIZE - 1))); do
    for j in $(seq 0 $((GRID_SIZE - 1))); do
        
        if [[ $SIGMOID == 'none' ]]; then
        # Directly sbatch the .slurm file while setting unique names and exporting variables
            sbatch --job-name="pv_${i}_${j}" \
                --output="logs/pvoros_${i}_${j}.out" \
                --error="logs/pvoros_${i}_${j}.err" \
                --export=ALL,IDX_I="$i",IDX_J="$j",GRID_SIZE="$GRID_SIZE" \
                run_pvoros.slurm
        elif [[ $SIGMOID == 'sigmoid' ]]; then
            sbatch --job-name="pv_${i}_${j}" \
                --output="logs/soft_pvoros_${i}_${j}.out" \
                --error="logs/soft_pvoros_${i}_${j}.err" \
                --export=ALL,IDX_I="$i",IDX_J="$j",GRID_SIZE="$GRID_SIZE" \
                run_soft_pv.slurm
        fi
    done
done