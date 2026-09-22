#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/jli48/slurmlog/err/log_%A.err
#SBATCH --output=/cluster/tufts/hugheslab/jli48/slurmlog/out/log_%A.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16g
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --job-name=cc_fraud

source ~/.bashrc

conda activate pvoros

python cc_fraud/train_cc_fraud.py

conda deactivate
