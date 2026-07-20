import numpy as np
import matplotlib.pyplot as plt

# Load your baseline structures
data = np.load('sweep_meta_data.npy', allow_pickle=True).item()
clfs = data['clfs']
train_test = data['train_test']

grid_size = 30
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for ax, seed in zip(axs.flat, clfs):
    w1_center, w2_center = clfs[seed].coef_[0, 0], clfs[seed].coef_[0, 1]
    intercept = float(clfs[seed].intercept_[0])
    
    w1_vals = np.linspace(w1_center - 2, w1_center + 2, grid_size)
    w2_vals = np.linspace(w2_center - 2, w2_center + 2, grid_size)
    
    # Empty canvas to rebuild the image matrix
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    
    # Read the data back matrix location by matrix location
    for i in range(grid_size):
        for j in range(grid_size):
            file_path = f"results_data_soft_pv/{seed}_res_{i}_{j}.txt"
            try:
                with open(file_path, 'r') as f:
                    heatmap[i, j] = float(f.read().strip())
            except FileNotFoundError:
                heatmap[i, j] = 0.0 # Fallback if a slurm node failed
                
    # --- Draw the Heatmap ---
    im = ax.imshow(
        heatmap, origin='lower',
        extent=[w1_vals[0], w1_vals[-1], w2_vals[0], w2_vals[-1]],
        aspect='auto', cmap='viridis'
    )
    fig.colorbar(im, ax=ax, label='pVOROS score')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_title(f'Soft Set pVOROS Matrix: {seed}')
    ax.scatter([w1_center], [w2_center], color='white', edgecolor='black', s=80, label='Trained')
    ax.legend(loc='upper right')

axs.flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig('soft_pvoros_surface_grid.pdf', format='pdf', dpi=300)
plt.show()