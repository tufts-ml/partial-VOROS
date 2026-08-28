#!/bin/bash
#SBATCH --error=logs/log_%A_%a.err
#SBATCH --output=logs/log_%A_%a.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16g
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --array=0-5

# source ~/.bashrc

# conda activate pvoros

python train_busi_pca.py \
    --dims="30" \
    --config_index="${SLURM_ARRAY_TASK_ID}"

# conda deactivate