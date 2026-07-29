import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

SEEDS = [
    'seed_101_201.npy',
    'seed_301_101.npy',
    'seed_501_801.npy',
    'seed_601_201.npy',
    'seed_701_501.npy'
]

# --- VOROS & SIGMOID PARAMETERS ---
KAPPA_FRAC = 0.5
ALPHA = 0.6
MIN_FP_COST_RATIO = 1/9
MAX_FP_COST_RATIO = 1/6
N_POINTS = 50
SIGMOID_K = 50

import _geometry_jax
import metrics_jax
import grad

if __name__ == "__main__":
    NUM_TRIALS = 10
    MAX_STEPS = 100
    LEARNING_RATE = 0.05
    grid_size = 30
    M = 1.0 

    # Load metadata dictionary
    data = np.load('heatmaps/sweep_meta_data.npy', allow_pickle=True).item()
    train_test = data['train_test']
    
    # Loss gradient setup using grad.py
    loss_and_grad = jax.value_and_grad(metrics_jax.pv_loss_theta_c)
    
    # Global grid space [-pi, pi] x [-3.0, 3.0]
    theta_vals = np.linspace(-np.pi, np.pi, grid_size)
    c_vals = np.linspace(-3.0, 3.0, grid_size)
    extent = [np.degrees(theta_vals[0]), np.degrees(theta_vals[-1]), c_vals[0], c_vals[-1]]
    thresholds = np.linspace(1e-5, 1.0-1e-5, 100)

    # DICTIONARY TO STORE OPTIMAL PARAMETERS ACROSS SEEDS
    optimal_params = {}

    # Iterate over all 5 seeds
    for seed in SEEDS:
        alpha = 0.6
        kappa_frac = 0.5
        print("\n" + "=" * 80)
        print(f"PROCESSING SEED: {seed}")
        print("=" * 80)
        
        # Match dataset used in heatmap generation
        X = np.asarray(train_test[seed][0])
        Y = np.asarray(train_test[seed][2])

        P = int(np.sum(Y == 1)) 
        N = int(np.sum(Y == 0)) 
        kappa = kappa_frac * (P + N)

        best_score = -np.inf
        best_trial_idx = -1
        all_trials_data = []

        np.random.seed(123)
        for trial in range(1, NUM_TRIALS + 1):
            theta_init = np.random.uniform(theta_vals[0], theta_vals[-1])
            c_init = np.random.uniform(c_vals[0], c_vals[-1])
            
            params = {
                'theta': jnp.array(theta_init, dtype=jnp.float64),
                'c': jnp.array(c_init, dtype=jnp.float64),
            }
            
            loss_history = []
            param_history = [(float(params['theta']), float(params['c']))]
            
            for step in range(1, MAX_STEPS + 1):
                loss_val, grads = loss_and_grad(params, X, Y, P, N, kappa, alpha, thresholds, MIN_FP_COST_RATIO, MAX_FP_COST_RATIO)
                
                params = {
                    'theta': params['theta'] - LEARNING_RATE * jnp.clip(grads['theta'], -1.0, 1.0),
                    'c': params['c'] - LEARNING_RATE * jnp.clip(grads['c'], -2.0, 2.0),
                }
                
                loss_history.append(float(loss_val))
                param_history.append((float(params['theta']), float(params['c'])))
                
            final_voros_score = -loss_history[-1]
            print(f"Trial {trial:2d} | Init: ({np.degrees(theta_init):.1f}°, {c_init:.2f}) | Final pVOROS: {final_voros_score:.6f}")
            
            trial_record = {
                'trial_num': trial,
                'param_history': np.array(param_history),
                'loss_history': loss_history,
                'final_score': final_voros_score
            }
            all_trials_data.append(trial_record)
            
            if final_voros_score > best_score:
                best_score = final_voros_score
                best_trial_idx = trial - 1

        print(f"\nSEED {seed} BEST: Trial {best_trial_idx + 1} with Score {best_score:.6f}")

        # --- SAVE OPTIMAL PARAMETERS FOR THIS SEED ---
        best_data = all_trials_data[best_trial_idx]
        optimal_theta = float(best_data['param_history'][-1, 0])
        optimal_c = float(best_data['param_history'][-1, 1])

        optimal_params[seed] = {
            'theta': optimal_theta,
            'c': optimal_c,
            'best_score': best_score,
            'trial_num': best_trial_idx + 1
        }

        # --- LOAD PRE-CALCULATED HEATMAP DATA ---
        heatmap = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                file_path = f"heatmaps/results_data/{seed}_res_{i}_{j}.txt"
                try:
                    with open(file_path, 'r') as f:
                        heatmap[i, j] = float(f.read().strip())
                except FileNotFoundError:
                    heatmap[i, j] = 0.0

        # --- PLOT OVERLAY ---
        plt.figure(figsize=(10, 7))
        im = plt.imshow(heatmap, origin='lower', extent=extent, aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im)
        cbar.set_label('Partial VOROS Score Metric', fontsize=11, labelpad=10)

        # 1. Plot all OTHER trajectories
        for idx, trial_data in enumerate(all_trials_data):
            if idx == best_trial_idx:
                continue
            
            t_raw = trial_data['param_history'][:, 0]
            t_norm = np.array([grad.wrap_to_pi(t) for t in t_raw])
            t_deg = np.degrees(t_norm)
            c_track = trial_data['param_history'][:, 1]

            plt.plot(t_deg, c_track, color='white', linestyle='-', linewidth=0.8, alpha=0.35, zorder=2)
            plt.scatter(t_deg[0], c_track[0], color='white', edgecolor='none', s=15, alpha=0.4, zorder=2)

        plt.plot([], [], color='white', linestyle='-', linewidth=1.0, alpha=0.5, label='Other Init Paths')

        # 2. Plot BEST trial trajectory
        best_raw_thetas = best_data['param_history'][:, 0]
        best_norm_thetas = np.array([grad.wrap_to_pi(t) for t in best_raw_thetas])
        best_thetas_deg = np.degrees(best_norm_thetas)
        best_cs_tracked = best_data['param_history'][:, 1]

        plt.plot(best_thetas_deg, best_cs_tracked, color='white', linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)
        plt.quiver(best_thetas_deg[:-1], best_cs_tracked[:-1], 
                   best_thetas_deg[1:] - best_thetas_deg[:-1], best_cs_tracked[1:] - best_cs_tracked[:-1], 
                   scale_units='xy', angles='xy', scale=1, color='red', width=0.0035, zorder=4, label='Best Gradient Steps')
        
        plt.scatter(best_thetas_deg[0], best_cs_tracked[0], color='#ff7f0e', edgecolor='black', s=55, zorder=5, label='Best Start')
        plt.scatter(best_thetas_deg[-1], best_cs_tracked[-1], color='#2ca02c', edgecolor='black', s=55, zorder=5, label='Best Convergence')

        plt.title(f'Gradient Descent Path: {seed} (Best: Trial {best_data["trial_num"]})\nHeld Constant Vector Norm ||w|| = 1.0', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Decision Boundary Angle (Degrees)', fontsize=11)
        plt.ylabel('Decision Boundary $y$-intercept ($c$)', fontsize=11)
        plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.4, color='white')
        
        plt.xlim(np.degrees(theta_vals[0]), np.degrees(theta_vals[-1]))
        plt.ylim(c_vals[0], c_vals[-1])
        
        plt.tight_layout()
        out_pdf = f"gradient_soft_pvoros_trajectory_{seed.replace('.npy', '')}.pdf"
        plt.savefig(out_pdf, format='pdf', dpi=300)
        plt.close()
        print(f"Saved plot: {out_pdf}")

    # --- SAVE ALL BEST OPTIMAL PARAMETERS TO AN OUTPUT FILE ---
    save_filepath = 'best_optimal_params.npy'
    np.save(save_filepath, optimal_params, allow_pickle=True)
    print("\n" + "=" * 80)
    print(f"SUCCESSFULLY SAVED BEST PARAMETERS TO: {save_filepath}")
    print("=" * 80)
    for seed, res in optimal_params.items():
        print(f"{seed:<20} | Theta*: {np.degrees(res['theta']):6.2f}° | c*: {res['c']:7.4f} | Best Score: {res['best_score']:.6f}")