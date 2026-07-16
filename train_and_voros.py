#!/usr/bin/env python3
"""Train logistic regression models on each seed and compute VOROS with gradients."""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import _geometry_jax
import _geometry
import time

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
KAPPA = 30
ALPHA = 0.2
# P and N are calculated from dataset prevalence
MIN_FP_COST_RATIO = 1/9
MAX_FP_COST_RATIO = 1/6
N_POINTS = 1000

# Sigmoid approximation parameters
SIGMOID_K = 50  # Steepness of sigmoid


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


def compute_smoothed_fprs_tprs(y_test, y_scores, thresholds):
    """Compute smoothed FPR and TPR using sigmoid approximation."""
    fprs_smooth = np.zeros(len(thresholds), dtype=float)
    tprs_smooth = np.zeros(len(thresholds), dtype=float)
    
    for t in range(1, len(thresholds)):
        tp, fp, tn, fn = soft_set_sigmoid(y_test, y_scores, tau=thresholds[t], k=SIGMOID_K)
        fprs_smooth[t] = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs_smooth[t] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return fprs_smooth, tprs_smooth


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
    grad_fn = jax.grad(voros_fn, argnums=(0, 1))
    grad_fprs, grad_tprs = grad_fn(fprs, tprs)
    grad_time = time.perf_counter() - start_time
    
    return voros_value, grad_fprs, grad_tprs, voros_time, grad_time


def compute_voros_numpy(fprs_np, tprs_np, thresholds, P, N):
    """Compute regular VOROS using NumPy implementation."""
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
    print(f"VOROS Parameters: κ={KAPPA}, α={ALPHA}")
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
    main()
