#!/usr/bin/env bash
#SBATCH --job-name=logisticreg$c     # Job name
#SBATCH --output=logs/logisticreg_%j.out        # Output file (%j is job ID, %x is job name)
#SBATCH --error=logs/logisticreg_%j.err          # Error file (%j is job ID, %x is job name)
#SBATCH -p batch,preempt
#SBATCH --time=02:00:00           # Time limit (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1       # CPU cores
#SBATCH --mem=8G                 # Memory per node


## INSERT YOUR PYTHON FILE THAT YOU WANT TO RUN AND RELEVANT CALLS
echo "Running weight decay = $w, step size = $s"
python -u train_busi_pca.py \
--w $w \
--s $s \
