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
SIGMOID_K = 50

def sigmoid_approximation(p, tau, k):
    """Sigmoid approximation: (1 + exp(-k * (p - tau)))^-1."""
    return (1 + np.exp(-k * (p - tau))) ** -1

def soft_set_sigmoid(y_true_N, y_pred_N, tau, k):
    """Compute soft TP, FP, TN, FN using sigmoid approximation."""
    y_true = np.asarray(y_true_N)
    y_pred = np.asarray(y_pred_N, dtype=float)
    soft_pred = sigmoid_approximation(y_pred, tau, k)
    
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    tp = soft_pred[pos_mask].sum()
    fn = (1 - soft_pred[pos_mask]).sum()
    fp = soft_pred[neg_mask].sum()
    tn = (1 - soft_pred[neg_mask]).sum()
    
    return tp, fp, tn, fn

def compute_smoothed_fprs_tprs(y_test, y_scores, thresholds):
    """Compute smoothed FPR and TPR using sigmoid approximation."""
    fprs_smooth = np.zeros(len(thresholds), dtype=float)
    tprs_smooth = np.zeros(len(thresholds), dtype=float)
    
    for t in range(1, len(thresholds)):
        tp, fp, tn, fn = soft_set_sigmoid(y_test, y_scores, tau=thresholds[t], k=SIGMOID_K)
        fprs_smooth[t] = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs_smooth[t] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return fprs_smooth, tprs_smooth


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
        
        eps = 1e-5
        thresholds = np.linspace(eps, 1.0 - eps, 100)
        fprs_smooth, tprs_smooth = compute_smoothed_fprs_tprs(y_true, y_pred, thresholds)
        

        _, acc_fprs, acc_tprs, _, satisfy = _geometry._kept_on_valid(
            fprs_smooth, tprs_smooth, thresholds, alpha, kappa, N, P
        )
        
        score = 0.0
        if satisfy:
            score = float(_geometry.voros(
                acc_fprs, acc_tprs, kappa, alpha, P, N,
                min_ratio, max_ratio, n_points=GRID_N_POINTS
            ))
            
        # Write results using the uniform grid position naming scheme
        out_name = f"results_data_soft_pv/{seed}_res_{args.idx_i}_{args.idx_j}.txt"
        with open(out_name, 'w') as f:
            f.write(f"{score}\n")

if __name__ == "__main__":
    main()