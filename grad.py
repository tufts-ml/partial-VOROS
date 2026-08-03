'''
contains relevant gradient functions

'''

import numpy as np
import jax
import jax.numpy as jnp
import _geometry_jax

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# --- VOROS & SIGMOID PARAMETERS ---
KAPPA_FRAC = 0.5
ALPHA = 0.6
MIN_FP_COST_RATIO = 1/9
MAX_FP_COST_RATIO = 1/6
N_POINTS = 50
SIGMOID_K = 50

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

def wrap_to_pi(theta):
    """Wraps any angle (in radians) strictly into [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

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
    
    # 3. Vectorized validity mask
    valid_mask = jax.vmap(
        lambda f, t: jnp.where(_geometry_jax.keep_model(f, t, ALPHA, KAPPA, N, P), 1.0, 0.0)
    )(fprs_smooth, tprs_smooth)
    
    satisfy = jnp.any(valid_mask > 0.0)
    
    # 4. Zero out invalid curve points pointwise
    acc_fprs = fprs_smooth * valid_mask
    acc_tprs = tprs_smooth * valid_mask
    
    # 5. Compute VOROS over the masked arrays
    voros_val = _geometry_jax.voros_jax(
        acc_fprs, acc_tprs, KAPPA, ALPHA, P, N,
        MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, n_points=N_POINTS
    )
    
    return -jnp.where(satisfy, voros_val, 0.0)




# --- In grad.py ---

# --- In grad.py ---

# def jax_voros_loss(params, x_val, y_val, P, N, M=1.0):
#     theta = params['theta']
#     c = params['c']
    
#     # 1. Map parameters back to linear boundary weights
#     w1 = M * jnp.sin(theta)
#     w2 = -M * jnp.cos(theta)
#     b = M * c * jnp.cos(theta)
    
#     w_vec = jnp.array([w1, w2])
#     logits = jnp.dot(x_val, w_vec) + b
#     y_scores = jax.nn.sigmoid(logits.ravel())
#     y_val_1d = y_val.ravel()

#     KAPPA = KAPPA_FRAC * (P + N)
#     eps = 1e-5
#     thresholds = jnp.linspace(eps, 1.0 - eps, 100)
    
#     # 2. Smooth ROC curve anchored at (0,0)
#     fprs_raw, tprs_raw = compute_smoothed_fprs_tprs_jax(y_val_1d, y_scores, thresholds)
#     fprs_smooth = fprs_raw.at[0].set(0.0)
#     tprs_smooth = tprs_raw.at[0].set(0.0)
    
#     # 3. Validity mask
#     valid_mask = jax.vmap(
#         lambda f, t: jnp.where(_geometry_jax.keep_model(f, t, ALPHA, KAPPA, N, P), 1.0, 0.0)
#     )(fprs_smooth, tprs_smooth)
    
#     # Check if ANY threshold satisfies the constraint
#     satisfy = jnp.any(valid_mask > 0.0)
    
#     # 4. Pointwise masked arrays
#     acc_fprs = fprs_smooth * valid_mask
#     acc_tprs = tprs_smooth * valid_mask
    
#     # 5. Compute VOROS
#     voros_val = _geometry_jax.voros_jax(
#         acc_fprs, acc_tprs, KAPPA, ALPHA, P, N,
#         MIN_FP_COST_RATIO, MAX_FP_COST_RATIO, n_points=N_POINTS
#     )
    
#     total_envelope_area, _ = _geometry_jax.total_region_area(P, N, ALPHA, KAPPA)
    
#     # 6. Guard with satisfy: If valid_mask is all zeros, return default envelope/zero
#     #    to match compute_reference_pvoros_kept_on_valid
#     safe_voros = jnp.where(satisfy, jnp.minimum(voros_val, total_envelope_area), total_envelope_area)
    
#     return -safe_voros