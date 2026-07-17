#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_curve

# --- ADJUSTED PATH INJECTION ---
# Find the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the PARENT directory (where _geometry actually lives)
parent_dir = os.path.dirname(script_dir) 
sys.path.append(parent_dir)

# Now Python can step out of the folder and find your extension flawlessly!
import _geometry

alpha = 0.6
kappa_frac = 0.5
min_ratio = 1/9
max_ratio = 1/6
GRID_N_POINTS = 50 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_i', type=int, required=True)
    parser.add_argument('--idx_j', type=int, required=True)
    parser.add_argument('--grid_size', type=int, required=True)
    args = parser.parse_args()
    
    # Load metadata array containing your classifiers
    metadata_path = os.path.join(script_dir, 'sweep_meta_data.npy')
    data = np.load(metadata_path, allow_pickle=True).item()
    clfs = data['clfs']
    train_test = data['train_test']
    
    # Process each seed using its unique coefficient center
    for seed in clfs.keys():
        y_true = np.asarray(train_test[seed][3])
        X_test = np.asarray(train_test[seed][1])
        
        P = int(np.sum(y_true == 1))
        N = int(np.sum(y_true == 0))
        kappa = kappa_frac * float(len(y_true))
        intercept = float(clfs[seed].intercept_[0])
        
        # DYNAMIC SEARCH CENTER TRACKING
        w1_center = clfs[seed].coef_[0, 0]
        w2_center = clfs[seed].coef_[0, 1]
        
        # Calculate the absolute target weight for this specific seed matrix
        w1_vals = np.linspace(w1_center - 2, w1_center + 2, args.grid_size)
        w2_vals = np.linspace(w2_center - 2, w2_center + 2, args.grid_size)
        
        w1 = w1_vals[args.idx_j]
        w2 = w2_vals[args.idx_i]
        
        # Standard fast geometric evaluation
        logits = X_test[:, 0] * w1 + X_test[:, 1] * w2 + intercept
        y_pred = 1.0 / (1.0 + np.exp(-logits))
        
        fprs, tprs, thrs = roc_curve(y_true, y_pred)
        
        _, acc_fprs, acc_tprs, _, satisfy = _geometry._kept_on_valid(
            fprs, tprs, thrs, alpha, kappa, N, P
        )
        
        score = 0.0
        if satisfy:
            score = float(_geometry.voros(
                acc_fprs, acc_tprs, kappa, alpha, P, N,
                min_ratio, max_ratio, n_points=GRID_N_POINTS
            ))
            
        # Write results using the uniform grid position naming scheme
        out_name = f"results_data/{seed}_res_{args.idx_i}_{args.idx_j}.txt"
        with open(out_name, 'w') as f:
            f.write(f"{score}\n")

if __name__ == "__main__":
    main()