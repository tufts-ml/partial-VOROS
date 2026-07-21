import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# --- VOROS & SIGMOID PARAMETERS ---
KAPPA_FRAC = 0.5
ALPHA = 0.6
MIN_FP_COST_RATIO = 1/9
MAX_FP_COST_RATIO = 1/6
N_POINTS = 50
SIGMOID_K = 50

import _geometry_jax

def sigmoid_jax(x, k=SIGMOID_K):
    return jax.nn.sigmoid(k * x)

def soft_set_sigmoid_jax(y_true_N, y_scores_N, tau, k):
    y_true = jnp.asarray(y_true_N, dtype=jnp.float64)
    y_scores = jnp.asarray(y_scores_N, dtype=jnp.float64)
    soft_pred = sigmoid_jax(y_scores - tau, k)

    pos_mask = y_true
    neg_mask = 1.0 - y_true

    tp = jnp.sum(soft_pred * pos_mask)
    fn = jnp.sum((1.0 - soft_pred) * pos_mask)
    fp = jnp.sum(soft_pred * neg_mask)
    tn = jnp.sum((1.0 - soft_pred) * neg_mask)

    return tp, fp, tn, fn

def compute_smoothed_fprs_tprs_jax(y_test, y_scores, thresholds):
    def one_threshold(tau):
        tp, fp, tn, fn = soft_set_sigmoid_jax(y_test, y_scores, tau, SIGMOID_K)
        tpr = tp / jnp.maximum(tp + fn, 1e-15)
        fpr = fp / jnp.maximum(fp + tn, 1e-15)
        return fpr, tpr

    return jax.vmap(one_threshold)(thresholds)

def jax_voros_loss(params, x_val, y_val, P, N, M=1.0):
    """JAX-tracable negative VOROS loss using fixed-shape pointwise masking."""
    theta = params['theta']
    c = params['c']
    
    # 1. Map parameters back to linear boundary weights
    w1 = M * jnp.sin(theta)
    w2 = -M * jnp.cos(theta)
    b = M * c * jnp.cos(theta)
    
    w_vec = jnp.array([w1, w2])
    logits = jnp.dot(x_val, w_vec) + b
    y_scores = jax.nn.sigmoid(logits.ravel())
    y_val_1d = y_val.ravel()

    KAPPA = KAPPA_FRAC * (P + N)
    eps = 1e-5
    thresholds = jnp.linspace(eps, 1.0 - eps, 100)
    
    # 2. Compute smooth ROC curve anchored at (0,0)
    fprs_raw, tprs_raw = compute_smoothed_fprs_tprs_jax(y_val_1d, y_scores, thresholds)
    fprs_smooth = fprs_raw.at[0].set(0.0)
    tprs_smooth = tprs_raw.at[0].set(0.0)
    
    # 3. Vectorized validity mask (returns 1.0 for valid points, 0.0 for invalid points)
    # This preserves array shape (100,) so JAX can trace it without errors!
    valid_mask = jax.vmap(
        lambda f, t: jnp.where(_geometry_jax.keep_model(f, t, ALPHA, KAPPA, N, P), 1.0, 0.0)
    )(fprs_smooth, tprs_smooth)
    
    satisfy = jnp.any(valid_mask > 0.0)
    
    # 4. Zero out invalid curve points pointwise without changing array shape
    acc_fprs = fprs_smooth * valid_mask
    acc_tprs = tprs_smooth * valid_mask
    
    # 5. Compute VOROS over the masked fixed-size arrays
    voros_val = _geometry_jax.voros_jax(
        acc_fprs, acc_tprs, KAPPA, ALPHA, P, N,
        MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, n_points=N_POINTS
    )
    
    return -jnp.where(satisfy, voros_val, 0.0)

def normalize_angle(theta, center_theta):
    """Normalize theta to lie within [center_theta - pi, center_theta + pi]."""
    return center_theta + (theta - center_theta + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    NUM_TRIALS = 10
    MAX_STEPS = 100
    LEARNING_RATE = 0.05
    grid_size = 30
    M = 1.0  # MATCHES HEATMAP CODE EXPLICITLY

    # Load metadata dictionary
    data = np.load('heatmaps/sweep_meta_data.npy', allow_pickle=True).item()
    clfs = data['clfs']
    train_test = data['train_test']
    
    seed = 'seed_501_801.npy'
    
    # Match dataset used in heatmap main() function
    X = np.asarray(train_test[seed][0])
    Y = np.asarray(train_test[seed][2])

    P = int(np.sum(Y == 1)) 
    N = int(np.sum(Y == 0)) 

    # Reconstruct original centers
    w1_center = float(clfs[seed].coef_[0, 0])
    w2_center = float(clfs[seed].coef_[0, 1])
    b_center = float(clfs[seed].intercept_[0])

    # Reconstruct grid angle and intercept reference points using M = 1
    theta_center = np.arctan2(w1_center, -w2_center)
    c_center = b_center / (M * np.cos(theta_center) + 1e-9)

    theta_vals = np.linspace(theta_center - np.radians(45), theta_center + np.radians(45), grid_size)
    c_vals = np.linspace(c_center - 2.0, c_center + 2.0, grid_size)

    # Gradient setup
    loss_and_grad = jax.value_and_grad(jax_voros_loss)
    best_score = -np.inf
    best_trial_data = None
    
    print("=" * 80)
    print(f"RUNNING {NUM_TRIALS} TRIALS MATCHING HEATMAP DATA (M={M})")
    print("=" * 80)
    
    np.random.seed(123)
    for trial in range(1, NUM_TRIALS + 1):
        # Sample starting points inside heatmap window
        theta_init = np.random.uniform(theta_vals[0], theta_vals[-1])
        c_init = np.random.uniform(c_vals[0], c_vals[-1])
        
        params = {
            'theta': jnp.array(theta_init, dtype=jnp.float64),
            'c': jnp.array(c_init, dtype=jnp.float64),
        }
        
        loss_history = []
        param_history = [(float(params['theta']), float(params['c']))]
        
        for step in range(1, MAX_STEPS + 1):
            loss_val, grads = loss_and_grad(params, X, Y, P, N, M)
            
            params = {
                'theta': params['theta'] - LEARNING_RATE * jnp.clip(grads['theta'], -1.0, 1.0),
                'c': params['c'] - LEARNING_RATE * jnp.clip(grads['c'], -2.0, 2.0),
            }
            
            loss_history.append(float(loss_val))
            param_history.append((float(params['theta']), float(params['c'])))
            
        final_voros_score = -loss_history[-1]
        print(f"Trial {trial:2d} | Init: ({np.degrees(theta_init):.1f}°, {c_init:.2f}) | Final pVOROS: {final_voros_score:.6f}")
        
        if final_voros_score > best_score:
            best_score = final_voros_score
            best_trial_data = {
                'trial_num': trial,
                'param_history': np.array(param_history),
                'loss_history': loss_history
            }

    print("\n" + "=" * 80)
    print(f"BEST PERFORMANCE: Trial {best_trial_data['trial_num']} with Score {best_score:.6f}")
    print("=" * 80)

    # --- LOAD PRE-CALCULATED HEATMAP DATA ---
    heatmap = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            file_path = f"heatmaps/results_data_soft_pv/{seed}_res_{i}_{j}.txt"
            try:
                with open(file_path, 'r') as f:
                    heatmap[i, j] = float(f.read().strip())
            except FileNotFoundError:
                heatmap[i, j] = 0.0

    # --- PLOT OVERLAY ---
    plt.figure(figsize=(10, 7))
    
    # Exact extent matching heatmap array dimensions
    extent = [np.degrees(theta_vals[0]), np.degrees(theta_vals[-1]), c_vals[0], c_vals[-1]]
    im = plt.imshow(heatmap, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    cbar = plt.colorbar(im)
    cbar.set_label('Partial VOROS Score Metric', fontsize=11, labelpad=10)
    
    # Extract, normalize, and project trajectory onto the heatmap axes
    raw_thetas = best_trial_data['param_history'][:, 0]
    norm_thetas = np.array([normalize_angle(t, theta_center) for t in raw_thetas])
    thetas_deg = np.degrees(norm_thetas)
    cs_tracked = best_trial_data['param_history'][:, 1]

    plt.plot(thetas_deg, cs_tracked, color='white', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
    plt.quiver(thetas_deg[:-1], cs_tracked[:-1], 
               thetas_deg[1:] - thetas_deg[:-1], cs_tracked[1:] - cs_tracked[:-1], 
               scale_units='xy', angles='xy', scale=1, color='red', width=0.004, zorder=3, label='Gradient Steps')
    
    plt.scatter(thetas_deg[0], cs_tracked[0], color='#ff7f0e', edgecolor='black', s=120, zorder=4, label='Start')
    plt.scatter(thetas_deg[-1], cs_tracked[-1], color='#2ca02c', edgecolor='black', s=120, zorder=4, label='Convergence')
    
    plt.title(f'Gradient Descent Path on Heatmap (Trial {best_trial_data["trial_num"]})\nHeld Constant Vector Norm ||w|| = 1.0', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Decision Boundary Angle $\theta$ (Degrees)', fontsize=11)
    plt.ylabel('Decision Boundary $y$-intercept ($c$)', fontsize=11)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.4, color='white')
    
    plt.xlim(np.degrees(theta_vals[0]), np.degrees(theta_vals[-1]))
    plt.ylim(c_vals[0], c_vals[-1])
    
    plt.tight_layout()
    plt.savefig('perfect_aligned_pvoros_trajectory.pdf', format='pdf', dpi=300)
    plt.show()