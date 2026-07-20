#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
GRID_SIZE = 30
FIXED_IDX_I = 5  # Choose the w2 index you want to freeze (0 to 29)
TARGET_SEED = 'seed_501_801.npy'  # Replace with one of your actual seed strings from clfs.keys()

# Paths to your results directories
DIR_HARD = "results_data"
DIR_SOFT = "results_data_soft_pv"
META_PATH = "sweep_meta_data.npy"  # Ensure this is in your running directory

def load_slice_data():
    # 1. Load metadata to reconstruct the exact same w1 values
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Could not find {META_PATH} to reconstruct weights.")
        
    data = np.load(META_PATH, allow_pickle=True).item()
    clfs = data['clfs']
    
    if TARGET_SEED not in clfs:
        available_seeds = list(clfs.keys())
        raise KeyError(f"Seed {TARGET_SEED} not found. Available seeds: {available_seeds}")
        
    # Reconstruct the weight array for w1 (idx_j) using the exact formula from your worker
    w1_center = clfs[TARGET_SEED].coef_[0, 0]
    w1_vals = np.linspace(w1_center - 2, w1_center + 2, GRID_SIZE)
    
    # Reconstruct fixed w2 value just for the plot title/labeling
    w2_center = clfs[TARGET_SEED].coef_[0, 1]
    w2_vals = np.linspace(w2_center - 2, w2_center + 2, GRID_SIZE)
    fixed_w2_val = w2_vals[FIXED_IDX_I]

    hard_scores = []
    soft_scores = []

    # 2. Iterate across all j indices (w1 range) for our fixed i index (w2)
    for j in range(GRID_SIZE):
        # Build file names matching your uniform naming scheme
        file_hard = os.path.join(DIR_HARD, f"{TARGET_SEED}_res_{FIXED_IDX_I}_{j}.txt")
        file_soft = os.path.join(DIR_SOFT, f"{TARGET_SEED}_res_{FIXED_IDX_I}_{j}.txt")
        
        # Read Hard PV score
        if os.path.exists(file_hard):
            with open(file_hard, 'r') as f:
                hard_scores.append(float(f.read().strip()))
        else:
            hard_scores.append(np.nan)  # Handle missing/failed jobs gracefully
            
        # Read Soft PV score
        if os.path.exists(file_soft):
            with open(file_soft, 'r') as f:
                soft_scores.append(float(f.read().strip()))
        else:
            soft_scores.append(np.nan)

    return w1_vals, np.array(hard_scores), np.array(soft_scores), fixed_w2_val

def main():
    try:
        w1_range, hard_y, soft_y, w2_val = load_slice_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Plotting
    plt.figure(figsize=(9, 5), dpi=120)
    
    plt.plot(w1_range, hard_y, label='pvoros (Hard Threshold)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(w1_range, soft_y, label='soft_pvoros (Sigmoid Approx)', color='#ff7f0e', linewidth=2, linestyle='--', marker='s', markersize=4)
    
    plt.title(f"PVoros vs Soft-PVoros Slice (Fixed $w_2$ = {w2_val:.4f} at idx={FIXED_IDX_I})", fontsize=12, pad=12)
    plt.xlabel("$w_1$ Weight Range", fontsize=11)
    plt.ylabel("Voros Score", fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    plt.tight_layout()
    
    # Save and show
    output_plot = f"slice_w2_idx_{FIXED_IDX_I}.png"
    plt.savefig(output_plot, bbox_inches='tight')
    print(f"Plot cleanly generated and saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()