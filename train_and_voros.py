#!/usr/bin/env python3
"""Train logistic regression models on each seed and compute VOROS with gradients."""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import _geometry_jax
import time

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
P = 10
N = 100
MIN_FP_COST_RATIO = 1/9
MAX_FP_COST_RATIO = 1/6
N_POINTS = 1000


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
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(x_train, y_train)
    
    # Get prediction scores on test set
    y_scores = model.predict_proba(x_test)[:, 1]
    
    # Compute FPR and TPR at different thresholds
    fprs, tprs, thresholds = roc_curve(y_test, y_scores)
    
    return fprs, tprs, thresholds, (x_train, x_test, y_train, y_test, model)


def compute_voros_with_grad(fprs_np, tprs_np):
    """Compute VOROS and its gradients w.r.t. fprs and tprs."""
    # Convert to JAX arrays
    fprs = jnp.array(fprs_np, dtype=jnp.float64)
    tprs = jnp.array(tprs_np, dtype=jnp.float64)
    
    # Define function for gradient computation
    def voros_fn(fprs, tprs):
        return _geometry_jax.voros_jax(
            fprs, tprs, KAPPA, ALPHA, P, N,
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


def main():
    """Process each seed: train model, compute VOROS, compute gradients."""
    print("=" * 80)
    print("LOGISTIC REGRESSION + VOROS GRADIENT COMPUTATION")
    print("=" * 80)
    print(f"VOROS Parameters: κ={KAPPA}, α={ALPHA}, P={P}, N={N}")
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
            fprs, tprs, thresholds, model_info = train_logistic_regression(x, y)
            train_time = time.perf_counter() - train_start
            print(f"  Training time: {train_time:.6f}s")
            print(f"  ROC points: {len(fprs)}")
            
            # Compute VOROS and gradients
            print("\nComputing VOROS and gradients...")
            voros_value, grad_fprs, grad_tprs, voros_time, grad_time = compute_voros_with_grad(fprs, tprs)
            
            # Store results
            results[seed_file] = {
                'voros': float(voros_value),
                'grad_fprs': np.array(grad_fprs),
                'grad_tprs': np.array(grad_tprs),
                'fprs': fprs,
                'tprs': tprs,
                'train_time': train_time,
                'voros_time': voros_time,
                'grad_time': grad_time,
            }
            
            # Print results
            print(f"  VOROS value: {voros_value:.8f}")
            print(f"  VOROS computation time: {voros_time:.6f}s")
            print(f"  Gradient computation time: {grad_time:.6f}s")
            print(f"\n  Gradient statistics:")
            print(f"    ∇_fprs: min={np.min(grad_fprs):.8f}, max={np.max(grad_fprs):.8f}, mean={np.mean(grad_fprs):.8f}")
            print(f"    ∇_tprs: min={np.min(grad_tprs):.8f}, max={np.max(grad_tprs):.8f}, mean={np.mean(grad_tprs):.8f}")
            print(f"    ∇_fprs L2 norm: {np.linalg.norm(grad_fprs):.8f}")
            print(f"    ∇_tprs L2 norm: {np.linalg.norm(grad_tprs):.8f}")
            
        except Exception as e:
            print(f"ERROR processing {seed_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Seed File':<25} {'VOROS':<15} {'Train (s)':<12} {'Comp (s)':<12} {'Grad (s)':<12}")
    print("-" * 80)
    
    for seed_file in SEEDS:
        if seed_file in results:
            res = results[seed_file]
            print(f"{seed_file:<25} {res['voros']:<15.8f} {res['train_time']:<12.6f} {res['voros_time']:<12.6f} {res['grad_time']:<12.6f}")
    
    print(f"\nTotal seeds processed: {len(results)}/{len(SEEDS)}")
    
    # Save results
    np.save('voros_results.npy', results, allow_pickle=True)
    print("\nResults saved to voros_results.npy")


if __name__ == "__main__":
    main()
