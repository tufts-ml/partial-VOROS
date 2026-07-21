import os
import numpy as np
import matplotlib.pyplot as plt

data = np.load('sweep_meta_data.npy', allow_pickle=True).item()
clfs = data['clfs']

grid_size = 30
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for ax, seed in zip(axs.flat, clfs):
    w1_center = clfs[seed].coef_[0, 0]
    w2_center = clfs[seed].coef_[0, 1]
    b_center = float(clfs[seed].intercept_[0])
    
    M = np.sqrt(w1_center**2 + w2_center**2)
    theta_center = np.arctan2(w1_center, -w2_center)
    c_center = b_center / (M * np.cos(theta_center) + 1e-9)
    
    # Matching the exact evaluation space geometry
    theta_vals = np.linspace(theta_center - np.radians(45), theta_center + np.radians(45), grid_size)
    c_vals = np.linspace(c_center - 2.0, c_center + 2.0, grid_size)
    
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    
    for i in range(grid_size):
        for j in range(grid_size):
            #### RESULTS FOLDER
            file_path = f"results_data/{seed}_res_{i}_{j}.txt"
            try:
                with open(file_path, 'r') as f:
                    heatmap[i, j] = float(f.read().strip())
            except FileNotFoundError:
                heatmap[i, j] = 0.0
                
    # Extent uses degrees for human readability on the X-axis
    extent = [np.degrees(theta_vals[0]), np.degrees(theta_vals[-1]), c_vals[0], c_vals[-1]]
    
    im = ax.imshow(
        heatmap, origin='lower',
        extent=extent,
        aspect='auto', cmap='viridis'
    )
    fig.colorbar(im, ax=ax, label='pVOROS score')
    ax.set_xlabel('Angle (Degrees)')
    ax.set_ylabel('y-intercept (c)')
    ax.set_title(f'True pVOROS Heatmap (magnitude = 1): {seed}')
    
    # Plot baseline anchor point
    ax.scatter([np.degrees(theta_center)], [c_center], color='white', edgecolor='black', s=80, label='Trained')
    ax.legend(loc='upper right')

axs.flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig('pvoros_angle_intercept_grid.pdf', format='pdf', dpi=300)
plt.show()