#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import _geometry_jax
import grad
from sklearn.metrics import roc_curve

# Fixed pVOROS Parameters
ALPHA = 0.6
KAPPA_FRAC = 0.5
MIN_RATIO = 1/9
MAX_RATIO = 1/6
GRID_N_POINTS = 50 
SIGMOID_K = 50

SEEDS = [
    'seed_101_201.npy',
    'seed_301_101.npy',
    'seed_501_801.npy',
    'seed_601_201.npy',
    'seed_701_501.npy'
]


def evaluate_soft_pvoros(x_train, y_true, theta, c, M):
    """Evaluate soft pVOROS score for a given boundary angle, intercept, and norm magnitude M."""
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    kappa = KAPPA_FRAC * float(len(y_true))
    
    # Scale linear weights by magnitude M
    w1 = M * np.sin(theta)
    w2 = -M * np.cos(theta)
    intercept = M * c * np.cos(theta)
    
    logits = x_train[:, 0] * w1 + x_train[:, 1] * w2 + intercept
    y_pred = 1.0 / (1.0 + np.exp(-logits))
    
    eps = 1e-5
    thresholds = np.linspace(eps, 1.0 - eps, 100)
    fprs_smooth, tprs_smooth = grad.compute_smoothed_fprs_tprs_jax(y_true, y_pred, thresholds)
    
    _, acc_fprs, acc_tprs, _, satisfy = _geometry_jax._kept_on_valid(
        fprs_smooth, tprs_smooth, thresholds, ALPHA, kappa, N, P
    )
    
    if satisfy:
        return float(_geometry_jax.voros_jax(
            acc_fprs, acc_tprs, kappa, ALPHA, P, N,
            MIN_RATIO, MAX_RATIO, n_points=GRID_N_POINTS
        ))
    return 0.0


def run_magnitude_experiment():
    # Range of magnitudes to sweep during testing (~0.1 to ~15.8)
    magnitudes = np.logspace(-1, 1.2, 50)

    # Load metadata and pre-calculated optimal parameters
    data = np.load('heatmaps/sweep_meta_data.npy', allow_pickle=True).item()
    train_test = data['train_test']
    
    optimal_params = np.load('best_optimal_params.npy', allow_pickle=True).item()
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    for idx, seed in enumerate(SEEDS):
        ax = axs.flat[idx]
        print("\n" + "=" * 80)
        print(f"TESTING MAGNITUDE FOR SEED: {seed}")
        print("=" * 80)
        
        X = np.asarray(train_test[seed][0])
        Y = np.asarray(train_test[seed][2])

        # Retrieve saved optimal parameters for this seed
        seed_data = optimal_params[seed]
        best_theta = seed_data['theta']
        best_c = seed_data['c']
        best_score_m1 = seed_data['best_score']

        print(f"Loaded Optimal Parameters:")
        print(f"  Theta*: {np.degrees(best_theta):.2f}° | c*: {best_c:.4f} | pVOROS (M=1): {best_score_m1:.6f}")

        # Evaluate soft pVOROS across the magnitude spectrum
        scores_vs_magnitude = []
        for M in magnitudes:

            w1 = M * np.sin(best_theta)
            w2 = -M * np.cos(best_c)
            intercept = M * best_c * np.cos(best_theta)
            
            # Fast geometric evaluation using the reparameterized elements
            logits = X[:, 0] * w1 + X[:, 1] * w2 + intercept
            y_pred = 1.0 / (1.0 + np.exp(-logits))
            
            fprs, tprs, thrs = roc_curve(Y, y_pred)

            P = int(np.sum(Y == 1))
            N = int(np.sum(Y == 0))
            kappa = KAPPA_FRAC * float(len(Y))
            
            _, acc_fprs, acc_tprs, _, satisfy = _geometry_jax._kept_on_valid(
                fprs, tprs, thrs, ALPHA, kappa, N, P
            )
            
            score = 0.0
            if satisfy:
                score = float(_geometry_jax.voros_jax(
                    acc_fprs, acc_tprs, kappa, ALPHA, P, N,
                    MIN_RATIO, MAX_RATIO, n_points=GRID_N_POINTS
                ))
            # score = evaluate_soft_pvoros(X, Y, best_theta, best_c, M)
            scores_vs_magnitude.append(score)

        # Plot magnitude curve on subplot
        ax.plot(magnitudes, scores_vs_magnitude, color='#1f77b4', linewidth=2, label='Soft pVOROS')
        
        # Highlight M=1 training baseline point
        ax.scatter([1.0], [best_score_m1], color='red', s=50, zorder=5, label='Training M = 1.0')
        
        # Plot formatting
        ax.set_xscale('log')
        ax.set_title(f'Seed: {seed.replace(".npy", "")}\n$(\\theta^*={np.degrees(best_theta):.1f}^\\circ, c^*={best_c:.2f})$', fontsize=11, fontweight='bold')
        ax.set_xlabel('Weight Magnitude $\|w\|$ (Log Scale)', fontsize=10)
        ax.set_ylabel('True pVOROS Score', fontsize=10)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)

    # Hide the empty 6th subplot
    axs.flat[-1].set_visible(False)
    
    plt.tight_layout()
    output_pdf = 'magnitude_grid_true_pvoros.pdf'
    plt.savefig(output_pdf, format='pdf', dpi=300)
    print("\n" + "=" * 80)
    print(f"Saved 2x3 grid plot to: {output_pdf}")
    print("=" * 80)
    plt.show()

if __name__ == "__main__":
    run_magnitude_experiment()