#!/bin/bash
#
# Usage
# -----
# $ bash train_busi_pca_grid.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi


## HYPERPARAMETERS YOU ARE GRID SEARCHING OVER 
declare -a weight_decay=(0.001 0.01 0.1 0)
declare -a num_iter=(25 50)
declare -a step_size=(0.0001 0.001)

# declare -a weight_decay=(1e-3)
# declare -a num_iter=(25)
# declare -a step_size=(0.00001)


for w in "${weight_decay[@]}"; do
    for i in "${num_iter[@]}"; do
        for s in "${step_size[@]}"; do
            export w=$w
            export i=$i
            export s=$s

            ## Use this line to see where you are in the loo
            ## Change to whatever you want to keep track of
            echo "Weight decay=$w  Iter=$i Step size=$s"

            ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

            if [[ $ACTION_NAME == 'submit' ]]; then
                ## Use this line to submit the experiment to the batch scheduler
                sbatch < train_busi_pca.slurm
            
            elif [[ $ACTION_NAME == 'run_here' ]]; then
                ## first run: srun -t 0-00:30 --mem 4000 -p interactive --pty bash

                ## Use this line to just run interactively
                bash train_busi_pca.slurm
            fi
        done
    done
done


