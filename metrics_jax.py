import numpy as np
from sklearn.metrics import roc_curve
import _geometry_jax
import _geometry
import jax.numpy as jnp

def pvoros_score_jax(y_true, y_pred, alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio,
                 n_points=1000):
    """Partial VOROS score with precision and capacity constraints.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels.
    y_pred : array-like of shape (n,)
        Predicted probabilities.
    alpha : float in (0, 1)
        Minimum precision (PPV) constraint.
    kappa_frac : float in (0, 1]
        Maximum predicted positive fraction (capacity = kappa_frac * len(y_true)).
    min_fp_cost_ratio : float
        Minimum C0/C1 cost ratio.
    max_fp_cost_ratio : float
        Maximum C0/C1 cost ratio.
    n_points : int
        Number of cost ratio grid points for integration.

    Returns
    -------
    float : pVOROS score in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    n = len(y_true)
    kappa = kappa_frac * float(n)

    fprs, tprs, _ = roc_curve(y_true, y_pred)
    
    # Cast to JAX arrays immediately
    j_fprs = jnp.asarray(fprs, dtype=jnp.float32)
    j_tprs = jnp.asarray(tprs, dtype=jnp.float32)
    
    # Fully vectorized mask creation on accelerator memory (No CPU loops!)
    upper_bound = (kappa - N * j_fprs) / P
    lower_bound = (alpha * N * j_fprs) / ((1 - alpha) * P)
    feasible = (j_tprs <= upper_bound) & (j_tprs >= lower_bound)

    return float(_geometry_jax.voros_jax(
        j_fprs, j_tprs, feasible, kappa, alpha, P, N,
        min_fp_cost_ratio, max_fp_cost_ratio, n_points=n_points,
    ))