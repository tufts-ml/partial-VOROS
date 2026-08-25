"""
Make heatmaps using results of w1w2_sweep.sh
"""

import os
import numpy as np
import matplotlib.pyplot as plt

data = np.load('sweep_meta_data.npy', allow_pickle=True).item()
clfs = data['clfs']

grid_size = 30
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Fixed global bounds matching main.py
theta_vals = np.linspace(-np.pi, np.pi, grid_size)
c_vals = np.linspace(-3.0, 3.0, grid_size)
extent = [np.degrees(theta_vals[0]), np.degrees(theta_vals[-1]), c_vals[0], c_vals[-1]]

for ax, seed in zip(axs.flat, clfs):
    # w1_center = float(clfs[seed].coef_[0, 0])
    # w2_center = float(clfs[seed].coef_[0, 1])
    # b_center = float(clfs[seed].intercept_[0])
    
    # Locate baseline classifier coordinates normalized to M = 1
    # theta_center = np.arctan2(w1_center, -w2_center)
    # c_center = b_center / (1.0 * np.cos(theta_center) + 1e-9)
    
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    
    for i in range(grid_size):
        for j in range(grid_size):
            file_path = f"results_data_soft_pv/{seed}_res_{i}_{j}.txt"
            try:
                with open(file_path, 'r') as f:
                    heatmap[i, j] = float(f.read().strip())
            except FileNotFoundError:
                heatmap[i, j] = 0.0
                
    im = ax.imshow(
        heatmap, origin='lower',
        extent=extent,
        aspect='auto', cmap='viridis'
    )
    fig.colorbar(im, ax=ax, label='pVOROS score')
    ax.set_xlabel('Angle (Degrees)')
    ax.set_ylabel('y-intercept (c)')
    ax.set_title(f'Soft pVOROS (Sigmoid K=50): {seed}')
    
    # # Plot baseline anchor point
    # ax.scatter([np.degrees(theta_center)], [c_center], color='white', edgecolor='black', s=80, label='Trained')
    # ax.legend(loc='upper right')

axs.flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig('soft_pv_k=50_npoints=1000.pdf', format='pdf', dpi=300)
plt.show()