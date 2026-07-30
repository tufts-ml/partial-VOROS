#!/usr/bin/env python3
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) 
sys.path.append(parent_dir)

import _geometry

alpha = 0.6
kappa_frac = 0.5
min_ratio = 1/9
max_ratio = 1/6
GRID_N_POINTS = 50 
SIGMOID_K = 10

def sigmoid_approximation(p, tau, k):
    return (1 + np.exp(-k * (p - tau))) ** -1

def soft_set_sigmoid(y_true_N, y_pred_N, tau, k):
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
    fprs_smooth = np.zeros(len(thresholds), dtype=float)
    tprs_smooth = np.zeros(len(thresholds), dtype=float)
    
    for t in range(1, len(thresholds)):
        tp, fp, tn, fn = soft_set_sigmoid(y_test, y_scores, tau=thresholds[t], k=SIGMOID_K)
        fprs_smooth[t] = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs_smooth[t] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return fprs_smooth, tprs_smooth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_i', type=int, required=True, help="Index for y-intercept grid")
    parser.add_argument('--idx_j', type=int, required=True, help="Index for angle grid")
    parser.add_argument('--grid_size', type=int, required=True)
    args = parser.parse_args()
    
    metadata_path = os.path.join(script_dir, 'sweep_meta_data.npy')
    data = np.load(metadata_path, allow_pickle=True).item()
    clfs = data['clfs']
    train_test = data['train_test']
    
    for seed in clfs.keys():
        y_true = np.asarray(train_test[seed][2])
        x_train = np.asarray(train_test[seed][0])
        
        P = int(np.sum(y_true == 1))
        N = int(np.sum(y_true == 0))
        kappa = kappa_frac * float(len(y_true))
        
        M = 1.0
        
        # FULL 360-DEGREE ANGLE SWEEP & INTERCEPT RANGE
        theta_vals = np.linspace(-np.pi, np.pi, args.grid_size)
        c_vals = np.linspace(-3.0, 3.0, args.grid_size)
        
        theta_curr = theta_vals[args.idx_j]
        c_curr = c_vals[args.idx_i]
        
        # Convert to line boundary parameters
        w1 = M * np.sin(theta_curr)
        w2 = -M * np.cos(theta_curr)
        intercept = M * c_curr * np.cos(theta_curr)
        
        logits = x_train[:, 0] * w1 + x_train[:, 1] * w2 + intercept
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
            
        out_name = f"results_data_soft_pv/{seed}_res_{args.idx_i}_{args.idx_j}.txt"
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        with open(out_name, 'w') as f:
            f.write(f"{score}\n")

if __name__ == "__main__":
    main()