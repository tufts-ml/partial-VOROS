## Computes grid of (soft and hard) pVOROS scores over grid of angle/intercept
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

        export i=$i
        export j=$j
        export GRID_SIZE=$GRID_SIZE
        
        if [[ $SIGMOID == 'none' ]]; then
        # Directly sbatch the .slurm file while setting unique names and exporting variables
            sbatch run_pvoros.slurm
        elif [[ $SIGMOID == 'sigmoid' ]]; then
            sbatch run_soft_pv.slurm
        fi
    done
done