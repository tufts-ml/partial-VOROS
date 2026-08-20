#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/jli48/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/tufts/hugheslab/jli48/slurmlog/out/log_%A_%a.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48g
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

source ~/.bashrc

conda activate pvoros

python train_busi_pca.py \
    --dims="30"

conda deactivate