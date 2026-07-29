import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from test_jax_loss import _theta_c_to_wb, _theta_c_to_wb_and_thresholds
from metrics_jax import compute_soft_roc, get_prediction_thresholds_dynamic, pv_loss
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

# def theta_c_to_wb(theta, c):
#     """Converts boundary angle (theta) and intercept (c) to normal vector w and bias b."""
#     w = jnp.array([jnp.sin(theta), -jnp.cos(theta)], dtype=jnp.float64)
#     b = c
#     return w, b

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
        print("\n" + "=" * 80)
        print(f"PROCESSING SEED: {seed}")
        print("=" * 80)
        
        X = np.asarray(train_test[seed][0])
        Y = np.asarray(train_test[seed][2])

        P = int(np.sum(Y == 1)) 
        N = int(np.sum(Y == 0)) 

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
                    # KAPPA_FRAC, ALPHA, thresholds, 
                    # MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, 
                    # N_POINTS, TEMP
                )
                
                # grad_w = grads_wb['w']
                # grad_b = grads_wb['b']
                
                # # Chain rule: convert dL/dw and dL/db -> dL/dtheta and dL/dc
                # grad_theta = grad_w[0] * jnp.cos(params['theta']) + grad_w[1] * jnp.sin(params['theta'])
                # grad_c = grad_b
                
                # params = {
                #     'theta': params['theta'] - LEARNING_RATE * jnp.clip(grad_theta, -1.0, 1.0),
                #     'c': params['c'] - LEARNING_RATE * jnp.clip(grad_c, -2.0, 2.0),
                # }

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