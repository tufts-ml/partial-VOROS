#!/usr/bin/env python3
"""Train logistic regression models on each seed and compute VOROS with gradients."""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import _geometry_jax
import _geometry
import time
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility of train-test split

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

SEEDS = [
    'seed_101_201.npy',
    'seed_301_101.npy',
    'seed_501_801.npy',
    'seed_601_201.npy',
    'seed_701_501.npy'
]

# VOROS parameters
KAPPA_FRAC = 0.5
ALPHA = 0.2
# P and N are calculated from dataset prevalence
MIN_FP_COST_RATIO = 1/9
MAX_FP_COST_RATIO = 1/6
N_POINTS = 1000

# Sigmoid approximation parameters
SIGMOID_K = 30  # Steepness of sigmoid


def sigmoid_approximation(p, tau, k):
    """Sigmoid approximation: (1 + exp(-k * (p - tau)))^-1."""
    return (1 + np.exp(-k * (p - tau))) ** -1


def soft_set_sigmoid(y_true_N, y_pred_N, tau, k):
    """Compute soft TP, FP, TN, FN using sigmoid approximation."""
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


def sigmoid_jax(x, k=SIGMOID_K):
    """JAX sigmoid approximation for soft thresholding."""
    return jax.nn.sigmoid(k * x)


def soft_set_sigmoid_jax(y_true_N, y_scores_N, tau, k):
    """Compute soft TP, FP, TN, FN using JAX-compatible sigmoid smoothing."""
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


def compute_smoothed_fprs_tprs(y_test, y_scores, thresholds):
    """Compute smoothed FPR and TPR using sigmoid approximation."""
    fprs_smooth = np.zeros(len(thresholds), dtype=float)
    tprs_smooth = np.zeros(len(thresholds), dtype=float)
    
    for t in range(1, len(thresholds)):
        tp, fp, tn, fn = soft_set_sigmoid(y_test, y_scores, tau=thresholds[t], k=SIGMOID_K)
        fprs_smooth[t] = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs_smooth[t] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return fprs_smooth, tprs_smooth


def compute_smoothed_fprs_tprs_jax(y_test, y_scores, thresholds):
    """Compute smoothed FPR and TPR using JAX and sigmoid approximation."""
    def one_threshold(tau):
        tp, fp, tn, fn = soft_set_sigmoid_jax(y_test, y_scores, tau, SIGMOID_K)
        tpr = tp / jnp.maximum(tp + fn, 1e-15)
        fpr = fp / jnp.maximum(fp + tn, 1e-15)
        return fpr, tpr

    fprs_smooth, tprs_smooth = jax.vmap(one_threshold)(thresholds)
    return fprs_smooth, tprs_smooth


def jax_voros_loss(params, x_val, y_val, P, N):  # Remove thresholds from signature
    """Negative VOROS loss with internal dynamic threshold grid scaling."""
    logits = jnp.dot(x_val, params['w']) + params['b']
    y_scores = jax.nn.sigmoid(logits)

    KAPPA = KAPPA_FRAC*(P+N)

    # Calculate prediction span dynamically inside the traced function
    eps = 1e-5
    thresholds = jnp.linspace(eps, 1.0 - eps, 100)
    
    fprs_smooth, tprs_smooth = compute_smoothed_fprs_tprs_jax(y_val, y_scores, thresholds)
    
    # Filter points
    _, acc_fprs, acc_tprs, _, satisfy = _geometry_jax._kept_on_valid(
        fprs_smooth, tprs_smooth, thresholds, ALPHA, KAPPA, N, P
    )
    
    # Calculate raw VOROS area
    voros_val = _geometry_jax.voros_jax(
        acc_fprs, acc_tprs, KAPPA, ALPHA, P, N,
        MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, n_points=N_POINTS
    )
    
    # CRITICAL: Cap the area at the true geometric bounds of your feasible region envelope
    total_envelope_area, _ = _geometry_jax.total_region_area(P, N, ALPHA, KAPPA)
    safe_voros = jnp.minimum(voros_val, total_envelope_area)
    
    # Zero out completely if the curve does not pass the filters
    final_voros = jnp.where(satisfy, safe_voros, 0.0)
    
    return -final_voros


def train_jax_voros_logistic(seed_filename, learning_rate=0.1, n_steps=100, n_thresholds=100, test_split=0.3):
    """Train a logistic regression model by minimizing negative JAX VOROS."""
    x, y = load_seed_data(seed_filename)
    n_samples = len(y)
    n_test = int(n_samples * test_split)
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Derive absolute counts from dataset dimensions
    P = int(np.sum(y_train)) # Validation positive count
    N = int(np.sum(1 - y_train)) # Validation negative count
    n_val = len(y_train)

    # Convert your fractional setting to an absolute integer count for the geometry engines
    # x_val_jax = jnp.asarray(x_val, dtype=jnp.float64)
    # y_val_jax = jnp.asarray(y_val, dtype=jnp.float64)
    eps = 1e-6
    # thresholds_jax = jnp.linspace(eps, 1.0 - eps, n_thresholds, dtype=jnp.float64)

    params = {
        'w': jnp.array((-2,-0.5), dtype=jnp.float64),
        'b': jnp.array(0.3, dtype=jnp.float64),
    }

    loss_and_grad = jax.value_and_grad(jax_voros_loss)
    history = []

    for step in range(1, n_steps + 1):
        # The loss function now handles its grid tracking internally and safely
        loss_val, grads = loss_and_grad(params, x_train, y_train, P, N)
        
        params = {
            'w': params['w'] - learning_rate * grads['w'],
            'b': params['b'] - learning_rate * grads['b'],
        }
        
        history.append((float(loss_val), float(jnp.linalg.norm(grads['w'])), float(abs(grads['b']))))
        if step % max(1, n_steps // 10) == 0 or step == 1:
            print(f"step={step:4d} loss={float(loss_val):.6f} w_norm={float(jnp.linalg.norm(params['w'])):.6f}")

    final_voros = -jax_voros_loss(params, x_test, y_test, P, N)
    return params, history, float(final_voros), P, N, x, y


def run_jax_gradient_descent_on_seed(seed_filename='seed_101_201.npy'):
    """Run JAX gradient descent on a single seed file."""
    print(f"Running JAX gradient descent with VOROS loss on {seed_filename}")
    params, history, voros_val, P, N, x_val_jax, y_val_jax = train_jax_voros_logistic(
        seed_filename, learning_rate=0.1, n_steps=100, n_thresholds=101, test_split=0.3
    )
    print(f"Final validation VOROS: {voros_val:.8f}")
    return params, history, voros_val, P, N, x_val_jax, y_val_jax


def load_seed_data(seed_filename):
    """Load data from a seed file."""
    data_dict = np.load(seed_filename, allow_pickle=True).item()
    x = data_dict['data']['x']
    y = data_dict['data']['y']
    return x, y


def train_logistic_regression(x, y, test_split=0.3):
    """Train logistic regression model and return fprs, tprs."""
    # Simple train-test split
    n_samples = len(y)
    n_test = int(n_samples * test_split)
    
    # Random split
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Calculate P and N from training set prevalence
    P = int(np.sum(y_train))
    N = int(np.sum(1 - y_train))
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(x_train, y_train)
    
    # Get prediction scores on test set
    y_scores = model.predict_proba(x_test)[:, 1]
    
    # Compute FPR and TPR at different thresholds
    fprs, tprs, thresholds = roc_curve(y_test, y_scores)
    
    return fprs, tprs, thresholds, (x_train, x_test, y_train, y_test, model), P, N


def compute_voros_with_grad(fprs_np, tprs_np, thresholds, P, N):
    """Compute VOROS and its gradients w.r.t. fprs and tprs."""
    # Convert to JAX arrays
    fprs = jnp.array(fprs_np, dtype=jnp.float64)
    tprs = jnp.array(tprs_np, dtype=jnp.float64)

    KAPPA = KAPPA_FRAC *(P+N)
    
    # Define function for gradient computation
    def voros_fn(fprs, tprs):
        _, acc_fprs, acc_tprs, _, _ = _geometry_jax._kept_on_valid(fprs, tprs, thresholds, ALPHA, KAPPA, N, P)
        return _geometry_jax.voros_jax(
            acc_fprs, acc_tprs, KAPPA, ALPHA, P, N,
            MIN_FP_COST_RATIO, MAX_FP_COST_RATIO,
            n_points=N_POINTS
        )
    
    # Compute VOROS
    start_time = time.perf_counter()
    voros_value = voros_fn(fprs, tprs)
    voros_time = time.perf_counter() - start_time
    
    # Compute gradients
    start_time = time.perf_counter()
    # grad_fn = jax.grad(voros_fn, argnums=(0, 1))
    # grad_fprs, grad_tprs = grad_fn(fprs, tprs)
    grad_fn = jax.value_and_grad(voros_fn, argnums=(0,1))
    loss, (grad_fprs, grad_tprs) = grad_fn(fprs, tprs)
    grad_time = time.perf_counter() - start_time
    
    return voros_value, grad_fprs, grad_tprs, voros_time, grad_time


def compute_voros_numpy(fprs_np, tprs_np, thresholds, P, N):
    """Compute regular VOROS using NumPy implementation."""
    KAPPA = KAPPA_FRAC *(P+N)
    # Define function for computation
    def voros_fn(fprs, tprs):
        _, acc_fprs, acc_tprs, _, _ = _geometry._kept_on_valid(fprs, tprs, thresholds, ALPHA, KAPPA, N, P)
        return _geometry.voros(
            acc_fprs, acc_tprs, KAPPA, ALPHA, P, N,
            MIN_FP_COST_RATIO, MAX_FP_COST_RATIO,
            n_points=N_POINTS
        )
    
    # Compute VOROS
    start_time = time.perf_counter()
    voros_value = voros_fn(fprs_np, tprs_np)
    voros_time = time.perf_counter() - start_time
    
    return voros_value, voros_time


def main():
    """Process each seed: train model, compute VOROS, compute gradients."""
    print("=" * 80)
    print("LOGISTIC REGRESSION + VOROS GRADIENT COMPUTATION")
    print("=" * 80)
    print(f"VOROS Parameters: κ_frac={KAPPA_FRAC}, α={ALPHA}")
    print(f"                  P and N are calculated from dataset prevalence")
    print(f"                  min_r={MIN_FP_COST_RATIO:.4f}, max_r={MAX_FP_COST_RATIO:.4f}")
    print(f"                  n_points={N_POINTS}\n")
    
    results = {}
    
    for seed_file in SEEDS:
        print(f"\n{'=' * 80}")
        print(f"Processing: {seed_file}")
        print(f"{'=' * 80}")
        
        try:
            # Load data
            x, y = load_seed_data(seed_file)
            print(f"Data shape: x={x.shape}, y={y.shape}")
            print(f"Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
            
            # Train model
            print("\nTraining logistic regression...")
            train_start = time.perf_counter()
            fprs, tprs, thresholds, model_info, P, N = train_logistic_regression(x, y)
            train_time = time.perf_counter() - train_start
            print(f"  Training time: {train_time:.6f}s")
            print(f"  ROC points: {len(fprs)}")
            print(f"  P (positives): {P}, N (negatives): {N}")
            
            # ===== NON-SMOOTH VERSION =====
            print("\n--- NON-SMOOTH VERSION ---")
            print("Computing VOROS (JAX) with original ROC curves...")
            voros_jax_nonsm, grad_fprs, grad_tprs, voros_time_nonsm, grad_time = compute_voros_with_grad(fprs, tprs, thresholds, P, N)
            
            print("Computing VOROS (NumPy) with original ROC curves...")
            voros_numpy_nonsm, voros_numpy_time_nonsm = compute_voros_numpy(fprs, tprs, thresholds, P, N)
            
            print(f"  VOROS (JAX):    {voros_jax_nonsm:.8f} ({voros_time_nonsm:.6f}s)")
            print(f"  VOROS (NumPy):  {voros_numpy_nonsm:.8f} ({voros_numpy_time_nonsm:.6f}s)")
            print(f"  Difference:     {abs(float(voros_jax_nonsm) - voros_numpy_nonsm):.10f}")
            
            # ===== SMOOTH VERSION =====
            print("\n--- SMOOTH VERSION (sigmoid approximation) ---")
            print("Computing smoothed curves with sigmoid approximation...")
            x_train, x_test, y_train, y_test, model = model_info
            y_scores = model.predict_proba(x_test)[:, 1]
            
            smooth_start = time.perf_counter()
            fprs_smooth, tprs_smooth = compute_smoothed_fprs_tprs(y_test, y_scores, thresholds)
            smooth_time = time.perf_counter() - smooth_start
            print(f"  Smoothing time: {smooth_time:.6f}s")
            
            print("Computing VOROS (JAX) with smoothed curves...")
            voros_jax_smooth, grad_fprs_smooth, grad_tprs_smooth, voros_time_smooth, grad_time_smooth = compute_voros_with_grad(fprs_smooth, tprs_smooth, thresholds, P, N)
            
            print("Computing VOROS (NumPy) with smoothed curves...")
            voros_numpy_smooth, voros_numpy_time_smooth = compute_voros_numpy(fprs_smooth, tprs_smooth, thresholds, P, N)
            
            print(f"  VOROS (JAX):    {voros_jax_smooth:.8f} ({voros_time_smooth:.6f}s)")
            print(f"  VOROS (NumPy):  {voros_numpy_smooth:.8f} ({voros_numpy_time_smooth:.6f}s)")
            print(f"  Difference:     {abs(float(voros_jax_smooth) - voros_numpy_smooth):.10f}")
            
            # Store results
            results[seed_file] = {
                # Non-smooth
                'voros_jax_nonsm': float(voros_jax_nonsm),
                'voros_numpy_nonsm': float(voros_numpy_nonsm),
                'fprs': fprs,
                'tprs': tprs,
                # Smooth
                'voros_jax_smooth': float(voros_jax_smooth),
                'voros_numpy_smooth': float(voros_numpy_smooth),
                'fprs_smooth': fprs_smooth,
                'tprs_smooth': tprs_smooth,
                # Gradients (from smooth version)
                'grad_fprs_smooth': np.array(grad_fprs_smooth),
                'grad_tprs_smooth': np.array(grad_tprs_smooth),
                # Timing
                'train_time': train_time,
                'voros_time_nonsm': voros_time_nonsm,
                'voros_numpy_time_nonsm': voros_numpy_time_nonsm,
                'voros_time_smooth': voros_time_smooth,
                'voros_numpy_time_smooth': voros_numpy_time_smooth,
                'smooth_time': smooth_time,
                'grad_time_smooth': grad_time_smooth,
                'P': P,
                'N': N,
            }
            
            # Print summary comparison
            print(f"\n--- COMPARISON ---")
            print(f"  JAX non-smooth:  {voros_jax_nonsm:.8f}")
            print(f"  JAX smooth:      {voros_jax_smooth:.8f}")
            print(f"  Difference:      {abs(voros_jax_smooth - voros_jax_nonsm):.10f}")
            print(f"\n  NumPy non-smooth: {voros_numpy_nonsm:.8f}")
            print(f"  NumPy smooth:     {voros_numpy_smooth:.8f}")
            print(f"  Difference:       {abs(voros_numpy_smooth - voros_numpy_nonsm):.10f}")
            print(f"\n  Gradient statistics (smooth):")
            print(f"    ∇_fprs: min={np.min(grad_fprs_smooth):.8f}, max={np.max(grad_fprs_smooth):.8f}, mean={np.mean(grad_fprs_smooth):.8f}")
            print(f"    ∇_tprs: min={np.min(grad_tprs_smooth):.8f}, max={np.max(grad_tprs_smooth):.8f}, mean={np.mean(grad_tprs_smooth):.8f}")
            print(f"    ∇_fprs L2 norm: {np.linalg.norm(grad_fprs_smooth):.8f}")
            print(f"    ∇_tprs L2 norm: {np.linalg.norm(grad_tprs_smooth):.8f}")
            
        except Exception as e:
            print(f"ERROR processing {seed_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY - NON-SMOOTH")
    print(f"{'=' * 80}")
    print(f"{'Seed File':<25} {'P':<8} {'N':<8} {'VOROS (JAX)':<15} {'VOROS (NumPy)':<15} {'Diff':<12}")
    print("-" * 100)
    
    for seed_file in SEEDS:
        if seed_file in results:
            res = results[seed_file]
            diff = abs(res['voros_jax_nonsm'] - res['voros_numpy_nonsm'])
            print(f"{seed_file:<25} {res['P']:<8} {res['N']:<8} {res['voros_jax_nonsm']:<15.8f} {res['voros_numpy_nonsm']:<15.8f} {diff:<12.10f}")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY - SMOOTH (sigmoid approximation)")
    print(f"{'=' * 80}")
    print(f"{'Seed File':<25} {'P':<8} {'N':<8} {'VOROS (JAX)':<15} {'VOROS (NumPy)':<15} {'Diff':<12}")
    print("-" * 100)
    
    for seed_file in SEEDS:
        if seed_file in results:
            res = results[seed_file]
            diff = abs(res['voros_jax_smooth'] - res['voros_numpy_smooth'])
            print(f"{seed_file:<25} {res['P']:<8} {res['N']:<8} {res['voros_jax_smooth']:<15.8f} {res['voros_numpy_smooth']:<15.8f} {diff:<12.10f}")
    
    print(f"\nTotal seeds processed: {len(results)}/{len(SEEDS)}")
    
    # Save results
    np.save('voros_results.npy', results, allow_pickle=True)
    print("\nResults saved to voros_results.npy")


if __name__ == "__main__":
    # main()
    params, history, voros_val, P, N, x_val_jax, y_val_jax = run_jax_gradient_descent_on_seed('seed_501_801.npy')
    
    print("\nFinal Optimization Status:")
    print(f"  Final Validation VOROS Score: {voros_val:.8f}")
    print(f"  Final Parameter Weights (w):  {params['w']}")
    print(f"  Final Parameter Bias (b):     {params['b']}")

    # 2. Extract loss trajectory from history elements (loss is index 0)
    # Since loss_val is stored as -VOROS, we keep it as-is to plot the loss minimization trend
    losses = [step_data[0] for step_data in history]
    iterations = np.arange(1, len(losses) + 1)

    # 3. Generate the Optimization Performance Plot
    plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot loss trajectory
    plt.plot(iterations, losses, color='#1f77b4', linewidth=2, label='Soft pVOROS Loss')
    # plt.scatter(iterations[0], losses[0], color='red', s=40, zorder=5, label='Initialization')
    # plt.scatter(iterations[-1], losses[-1], color='green', s=40, zorder=5, label='Optimized Convergence')

    # Formatting and labels
    plt.title('JAX Gradient Descent: pVOROS Loss Trajectory (Seed 501_801)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Gradient Descent Iteration Number', fontsize=10)
    plt.ylabel('Loss Value (Negative Partial VOROS Score)', fontsize=10)
    plt.xlim(0, len(losses) + 1)
    
    # Force y-axis limit safety constraints based on calculated metrics
    min_loss, max_loss = min(losses), max(losses)
    plt.ylim(min_loss - 0.05, max_loss + 0.05)
    
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()

    # 4. Save to directory as a vector PDF
    output_pdf_path = 'pvoros_loss_trajectory_seed_501_801.pdf'
    plt.savefig(output_pdf_path, format='pdf', dpi=300)
    plt.close()