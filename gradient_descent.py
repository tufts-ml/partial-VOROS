import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from test_jax_loss import _theta_c_to_wb
from metrics_jax import compute_soft_roc
import grad

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
TEMP = 0.03

def compute_decision_scores(X, theta, c):
    """Computes continuous decision scores w · x + b."""
    w, b = _theta_c_to_wb(theta, c)
    raw_scores = jnp.dot(X, w) + b
    return jax.nn.sigmoid(raw_scores)

if __name__ == "__main__":
    NUM_TRIALS = 10
    MAX_STEPS = 100
    LEARNING_RATE = 0.05
    grid_size = 30

    # Load metadata dictionary
    data = np.load('heatmaps/sweep_meta_data.npy', allow_pickle=True).item()
    train_test = data['train_test']
    
    # Loss gradient setup wrt 'w' and 'b' using metrics_jax.pv_loss
    loss_and_grad_wb = jax.value_and_grad(grad.jax_voros_loss)
    
    theta_vals = np.linspace(-np.pi, np.pi, grid_size)
    c_vals = np.linspace(-3.0, 3.0, grid_size)

    optimal_params = {}

    for seed in SEEDS:
        alpha = 0.6
        kappa_frac = 0.5
        print("\n" + "=" * 80)
        print(f"PROCESSING SEED: {seed}")
        print("=" * 80)
        
        X = np.asarray(train_test[seed][0])
        Y = np.asarray(train_test[seed][2])

        P = int(np.sum(Y == 1)) 
        N = int(np.sum(Y == 0)) 
        print(f"prevalence = {(P/(P+N)):.3f}")
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
                w_vec, b_val = _theta_c_to_wb(params['theta'], params['c'])
                wb_params = {'w': w_vec, 'b': b_val}
                
                # Compute loss and gradients using metrics_jax.pv_loss
                loss_val, grads = loss_and_grad_wb(
                    params, X, Y, P, N, 1.0
                )

                params = {
                    'theta': params['theta'] - LEARNING_RATE * jnp.clip(grads['theta'], -1.0, 1.0),
                    'c': params['c'] - LEARNING_RATE * jnp.clip(grads['c'], -2.0, 2.0),
                }
                
                loss_history.append(float(loss_val))
                param_history.append((float(params['theta']), float(params['c'])))
                
            final_voros_score = -loss_history[-1]
            
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
        im = plt.imshow(heatmap, origin='lower', aspect='auto', cmap='viridis')
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
        # ---------------------------------------------------------
        # GENERATE ROC CURVE + ALPHA & KAPPA CONSTRAINTS PLOT
        # ---------------------------------------------------------
        scores = compute_decision_scores(X, optimal_theta, optimal_c)
        fprs_raw, tprs_raw, _ = compute_soft_roc(Y, scores, temp=TEMP)
        fprs_smooth = fprs_raw.at[0].set(0.0)
        tprs_smooth = tprs_raw.at[0].set(0.0)
        roc_auc = auc(fprs_smooth, tprs_smooth)

        plt.figure(figsize=(8, 8))
        
        # 1. Plot empirical ROC curve
        plt.plot(fprs_smooth, tprs_smooth, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance Baseline')

        # 2. Alpha (Precision) Constraint Boundary
        prevalence = P / (P + N)
        alpha_slope = ALPHA * (1 - prevalence) / (prevalence * (1 - ALPHA))
        
        fpr_grid = np.linspace(0, 1, 200)
        tpr_alpha_bound = alpha_slope * fpr_grid

        plt.plot(
            fpr_grid, tpr_alpha_bound, color='#d62728', linestyle='--', lw=2.0, 
            label=f'Precision Boundary $\\alpha$ (Slope = {alpha_slope:.2f})'
        )

        # 3. Kappa (Capacity / Alarm Budget) Constraint Boundary
        kappa = KAPPA_FRAC * (P + N)
        kappa_slope = -(N / P)
        tpr_kappa_bound = kappa_slope * fpr_grid + (kappa / P)

        plt.plot(
            fpr_grid, tpr_kappa_bound, color='#2ca02c', linestyle='--', lw=2.0, 
            label=f'Capacity Boundary $\\kappa$ (Slope = {kappa_slope:.2f})'
        )

        # 4. Highlight the Feasible Operating Region
        # Clamp both lower and upper bounds strictly to the [0, 1] ROC square range
        y_lower_clamped = np.clip(tpr_alpha_bound, 0.0, 1.0)
        y_upper_clamped = np.clip(tpr_kappa_bound, 0.0, 1.0)

        # Fill region where upper bound strictly exceeds lower bound
        plt.fill_between(
            fpr_grid, 
            y_lower_clamped, 
            y_upper_clamped, 
            where=(y_upper_clamped >= y_lower_clamped),
            color='#ff7f0e', 
            alpha=0.18, 
            label='Feasible Region'
        )

        # Formatting
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(
            f'ROC Curve with Dual Slope Bounds ($\\alpha$ & $\\kappa$): {seed}\n'
            f'Optimal $\\theta$: {np.degrees(optimal_theta):.1f}°, $c$: {optimal_c:.3f}', 
            fontsize=13, fontweight='bold', pad=12
        )
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        
        plt.tight_layout()
        out_roc_pdf = f"roc_dual_slope_bounded_{seed.replace('.npy', '')}.pdf"
        plt.savefig(out_roc_pdf, format='pdf', dpi=300)
        plt.close()
        print(f"Saved plot: {out_roc_pdf}")

    # Save optimal parameters
    save_filepath = 'best_optimal_params.npy'
    np.save(save_filepath, optimal_params, allow_pickle=True)
    print("\n" + "=" * 80)
    print(f"SUCCESSFULLY SAVED BEST PARAMETERS TO: {save_filepath}")
    print("=" * 80)