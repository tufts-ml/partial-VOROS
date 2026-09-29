#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/jli48/slurmlog/err/log_%A_%a.err
#SBATCH --output=/cluster/tufts/hugheslab/jli48/slurmlog/out/log_%A_%a.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16g
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --job-name=cc_fraud
#SBATCH --array=0-5

source ~/.bashrc

conda activate pvoros

# 3 alpha/kappa_frac configs x 2 min_fp/max_fp configs = 6 independent jobs,
# indexed by SLURM_ARRAY_TASK_ID (0-5).
ALPHA_VALS=(0.5 0.1 0.5)
KAPPA_VALS=(1.0 0.5 0.5)
MIN_FP_VALS=(0.1111111111111111 6)
MAX_FP_VALS=(0.16666666666666666 9)

AK_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
FP_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

ALPHA=${ALPHA_VALS[$AK_IDX]}
KAPPA_FRAC=${KAPPA_VALS[$AK_IDX]}
MIN_FP=${MIN_FP_VALS[$FP_IDX]}
MAX_FP=${MAX_FP_VALS[$FP_IDX]}

python cc_fraud/train_cc_fraud.py \
    --alpha "$ALPHA" \
    --kappa_frac "$KAPPA_FRAC" \
    --min_fp "$MIN_FP" \
    --max_fp "$MAX_FP"

conda deactivate
