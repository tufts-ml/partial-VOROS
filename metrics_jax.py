import numpy as np
from sklearn.metrics import roc_curve
import _geometry_jax
import jax.numpy as jnp
import jax

def voros_score(y_true, y_pred, min_fp_cost_ratio, max_fp_cost_ratio,
                 n_points=1000):
    """Partial VOROS score with precision and capacity constraints.

    Returns
    -------
    float : pVOROS score in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    n = len(y_true)

    if P == 0 or N == 0:
        return 0.0

    fprs, tprs, _ = roc_curve(y_true, y_pred)
    
    # Cast to JAX arrays
    j_fprs = jnp.asarray(fprs, dtype=jnp.float64)
    j_tprs = jnp.asarray(tprs, dtype=jnp.float64)

    voros_val = _geometry_jax.voros_jax(
        j_fprs, 
        j_tprs, 
        float(P + N), 
        1e-8, 
        P, 
        N,
        min_fp_cost_ratio, 
        max_fp_cost_ratio, 
        n_points
    )

    return float(voros_val)

def pvoros_score(y_true, y_pred, alpha, kappa_frac, min_fp_cost_ratio, max_fp_cost_ratio,
                 n_points=1000):
    """Partial VOROS score with precision and capacity constraints.

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

    if P == 0 or N == 0:
        return 0.0

    fprs, tprs, _ = roc_curve(y_true, y_pred)
    
    # Cast to JAX arrays
    j_fprs = jnp.asarray(fprs, dtype=jnp.float64)
    j_tprs = jnp.asarray(tprs, dtype=jnp.float64)
    
    # Compute feasibility bounds with numerical tolerance guardrails
    upper_bound = (kappa - N * j_fprs) / float(P)
    lower_bound = (alpha * N * j_fprs) / float((1.0 - alpha) * P)
    feasible = (j_tprs <= upper_bound + 1e-9) & (j_tprs >= lower_bound - 1e-9)

    # If no ROC points satisfy constraints, return 0.0 score immediately
    if not jnp.any(feasible):
        return 0.0

    # Filter ROC points to only keep feasible operating points
    acc_fprs = j_fprs[feasible]
    acc_tprs = j_tprs[feasible]

    voros_val = _geometry_jax.voros_jax(
        acc_fprs, 
        acc_tprs, 
        kappa, 
        alpha, 
        P, 
        N,
        min_fp_cost_ratio, 
        max_fp_cost_ratio, 
        n_points
    )

    return float(voros_val)

def prediction_thresholds_fixed():
    """Fixed thresholds for differentiable evaluation."""
    eps = 1e-5
    thresholds = jnp.linspace(eps, 1.0 - eps, 100)

    return thresholds

def soft_roc_fixed_thresholds(y_true, y_pred, temp=0.02):
    """Computes a differentiable soft approximation of FPR and TPR using fixed
    thresholds.
    
    Args:
        y_true: Binary ground truth labels of shape (N,)
        y_pred: Predicted probabilities of shape (N,)
        temp: Temperature parameter. Smaller values get closer to the real 
              step function, but make gradients sharper/harder to optimize.
    """
    # Reshape for broadcasting: (N, 1) and (1, M)
    y_true_col = y_true[:, None]
    y_pred_col = y_pred[:, None]
    thresholds = prediction_thresholds_fixed()
    thresh_row = thresholds[None, :]
    
    # Soft approximation of indicator I(y_pred >= threshold)
    soft_indicators = jax.nn.sigmoid((y_pred_col - thresh_row) / temp)
    
    # Target class masks
    pos_mask = y_true_col
    neg_mask = 1.0 - y_true_col
    
    # Compute soft True Positives and False Positives
    soft_tps = jnp.sum(soft_indicators * pos_mask, axis=0)
    soft_fps = jnp.sum(soft_indicators * neg_mask, axis=0)
    
    # Actual positive/negative counts (safe division)
    P = jnp.maximum(jnp.sum(pos_mask), 1e-5)
    N = jnp.maximum(jnp.sum(neg_mask), 1e-5)
    
    # Output soft curves
    soft_tprs = soft_tps / P
    soft_fprs = soft_fps / N

    # soft_tprs = jnp.clip(soft_tps / P, 0.0, 1.0)
    # soft_fprs = jnp.clip(soft_fps / N, 0.0, 1.0)
    
    return soft_fprs, soft_tprs, thresholds

def prediction_thresholds_dynamic(y_pred, num_thresholds=1000):
    """Dynamic quantile thresholds for non-differentiable evaluation."""
    eps = 1e-5
    q = jnp.linspace(1.0 - eps, eps, num_thresholds)
    thresholds = jnp.quantile(y_pred, q)
    return jax.lax.stop_gradient(thresholds)

def soft_roc_dynamic_thresholds(y_true, y_pred, temp=0.02):
    """Computes a differentiable soft approximation of FPR and TPR using
    dynamic quantile thresholds.
    
    Args:
        y_true: Binary ground truth labels of shape (N,)
        y_pred: Predicted probabilities of shape (N,)
        temp: Temperature parameter. Smaller values get closer to the real 
              step function, but make gradients sharper/harder to optimize.
    """
    # Reshape for broadcasting: (N, 1) and (1, M)
    y_true_col = y_true[:, None]
    y_pred_col = y_pred[:, None]
    thresholds = prediction_thresholds_dynamic(y_pred)
    thresh_row = thresholds[None, :]
    
    # Soft approximation of indicator I(y_pred >= threshold)
    soft_indicators = jax.nn.sigmoid((y_pred_col - thresh_row) / temp)
    
    # Target class masks
    pos_mask = y_true_col
    neg_mask = 1.0 - y_true_col
    
    # Compute soft True Positives and False Positives
    soft_tps = jnp.sum(soft_indicators * pos_mask, axis=0)
    soft_fps = jnp.sum(soft_indicators * neg_mask, axis=0)
    
    # Actual positive/negative counts (safe division)
    P = jnp.maximum(jnp.sum(pos_mask), 1e-5)
    N = jnp.maximum(jnp.sum(neg_mask), 1e-5)
    
    # Output soft curves
    soft_tprs = soft_tps / P
    soft_fprs = soft_fps / N


    # soft_tprs = jnp.clip(soft_tps / P, 0.0, 1.0)
    # soft_fprs = jnp.clip(soft_fps / N, 0.0, 1.0)
    
    return soft_fprs, soft_tprs, thresholds

def pv_loss(
    params, 
    X, 
    y_true, 
    P, 
    N,
    kappa, 
    alpha,
    min_fp_cost_ratio, 
    max_fp_cost_ratio, 
    n_points=1000, 
    temp=0.02):
    """JAX-tracable negative VOROS loss using fixed-shape pointwise masking."""
    theta = params.get("theta", params.get("w"))
    c = params.get("c", params.get("b"))

    # 1. Map parameters back to linear boundary weights
    
    logits = jnp.dot(X, theta) + c
    y_pred = jax.nn.sigmoid(logits)
    
    # 2. Compute smooth ROC curve anchored at (0,0)
    fprs_raw, tprs_raw, _ = soft_roc_dynamic_thresholds(y_true, y_pred,temp=temp)
    fprs_smooth = fprs_raw.at[0].set(0.0)
    tprs_smooth = tprs_raw.at[0].set(0.0)
    
    # 3. Vectorized validity mask (returns 1.0 for valid points, 0.0 for invalid points)
    # This preserves array shape (100,) so JAX can trace it without errors!
    valid_mask = jax.vmap(
        lambda f, t: jnp.where(_geometry_jax.keep_model(f, t, alpha, kappa, N, P), 1.0, 0.0)
    )(fprs_smooth, tprs_smooth)
    
    satisfy = jnp.any(valid_mask > 0.0)
    
    # 4. Zero out invalid curve points pointwise without changing array shape
    acc_fprs = fprs_smooth * valid_mask
    acc_tprs = tprs_smooth * valid_mask
    
    # 5. Compute VOROS over the masked fixed-size arrays
    voros_val = _geometry_jax.voros_jax(
        acc_fprs, 
        acc_tprs, 
        kappa, 
        alpha, 
        P, 
        N,
        min_fp_cost_ratio, 
        max_fp_cost_ratio, 
        n_points
    )
    
    return -jnp.where(satisfy, voros_val, 0.0)

def pv_loss_fixed_thresh(
    params, 
    X, 
    y, 
    P, 
    N,
    kappa, 
    alpha,
    min_fp_cost_ratio, 
    max_fp_cost_ratio, 
    n_points=1000, 
    temp=0.02):
    """JAX-tracable negative VOROS loss using fixed-shape pointwise masking."""
    theta = params.get("theta", params.get("w"))
    c = params.get("c", params.get("b"))

    # 1. Map parameters back to linear boundary weights
    
    logits = jnp.dot(X, theta) + c
    y_scores = jax.nn.sigmoid(logits.ravel())
    y_val_1d = y_scores.ravel()
    
    # 2. Compute smooth ROC curve anchored at (0,0)
    fprs_raw, tprs_raw, _ = soft_roc_fixed_thresholds(y, y_val_1d, temp=temp)
    fprs_smooth = fprs_raw.at[0].set(0.0)
    tprs_smooth = tprs_raw.at[0].set(0.0)
    
    # 3. Vectorized validity mask
    valid_mask = jax.vmap(
        lambda f, t: jnp.where(_geometry_jax.keep_model(f, t, alpha, kappa, N, P), 1.0, 0.0)
    )(fprs_smooth, tprs_smooth)
    
    satisfy = jnp.any(valid_mask > 0.0)
    
    # 4. Zero out invalid curve points pointwise
    acc_fprs = fprs_smooth * valid_mask
    acc_tprs = tprs_smooth * valid_mask
    
    # 5. Compute VOROS over the masked arrays
    voros_val = _geometry_jax.voros_jax(
        acc_fprs, 
        acc_tprs, 
        kappa, 
        alpha, 
        P, 
        N,
        min_fp_cost_ratio, 
        max_fp_cost_ratio, 
        n_points=n_points)
    
    return -jnp.where(satisfy, voros_val, 0.0)

def pvoros_loss_kept_on_valid(
    params, 
    X, 
    y_true, 
    kappa, 
    alpha,
    min_fp_cost_ratio, 
    max_fp_cost_ratio, 
    n_points=1000, 
    temp=0.02):
    """Differentiable Partial VOROS loss function."""
    P = jnp.sum(y_true == 1)
    N = jnp.sum(y_true == 0)

    w = params['w']
    b = params['b']
    logits = jnp.dot(X, w) + b
    y_pred = jax.nn.sigmoid(logits)
    
    fprs, tprs, thresholds = soft_roc_fixed_thresholds(y_true, y_pred, temp=temp)
    _, acc_fprs, acc_tprs, _, satisfy = _geometry_jax._kept_on_valid(fprs, tprs, thresholds, alpha, kappa, N, P)

    ## 3. Call JAX-compatible VOROS function
    vor = _geometry_jax.voros_jax(
        fprs=acc_fprs,
        tprs=acc_tprs,
        κ=kappa,
        α=alpha,
        P=P,
        N=N,
        min_fp_cost_ratio=min_fp_cost_ratio,  
        max_fp_cost_ratio=max_fp_cost_ratio,
        n_points=n_points,           
        thresholds=thresholds  # Must pass your defined array of thresholds here
    )

    return -vor, satisfy


def pv_loss_theta_c(
    params, 
    X, 
    y_true, 
    P,
    N,
    kappa, 
    alpha,
    min_fp_cost_ratio, 
    max_fp_cost_ratio, 
    n_points=1000, 
    temp=0.02,
    M=1.0):
    """Differentiable Partial VOROS loss function."""
    theta = params['theta']
    c = params['c']

    w1 = M * jnp.sin(theta)
    w2 = -M * jnp.cos(theta)
    b = M * c * jnp.cos(theta)
    w = jnp.array([w1,w2])

    # 1. Forward Pass
    logits = jnp.dot(X, w) + b
    y_pred = jax.nn.sigmoid(logits)

    eps = 1e-5
    thresholds = jnp.linspace(eps, 1.0 - eps, 100)
        
    # 2. Compute smooth ROC curve anchored at (0,0)
    fprs_raw, tprs_raw, _ = soft_roc_fixed_thresholds(y_true, y_pred, temp=temp)
    fprs_smooth = fprs_raw.at[0].set(0.0)
    tprs_smooth = tprs_raw.at[0].set(0.0)
    
    # 3. Vectorized validity mask (returns 1.0 for valid points, 0.0 for invalid points)
    # This preserves array shape (100,) so JAX can trace it without errors!
    valid_mask = jax.vmap(
        lambda f, t: jnp.where(_geometry_jax.keep_model(f, t, alpha, kappa, N, P), 1.0, 0.0)
    )(fprs_smooth, tprs_smooth)
    
    satisfy = jnp.any(valid_mask > 0.0)
    
    # 4. Zero out invalid curve points pointwise without changing array shape
    acc_fprs = fprs_smooth * valid_mask
    acc_tprs = tprs_smooth * valid_mask
    
    # 5. Compute VOROS over the masked fixed-size arrays
    voros_val = _geometry_jax.voros_jax(
        acc_fprs, 
        acc_tprs, 
        kappa, 
        alpha, 
        P, 
        N,
        min_fp_cost_ratio, 
        max_fp_cost_ratio, 
        n_points
    )

    # total_envelope_area, _ = _geometry_jax.total_region_area(P, N, 0.6, kappa)
    # env_area_scalar = float(np.asarray(total_envelope_area).item())
    # voros_val = min(voros_val, env_area_scalar)
    
    return -jnp.where(satisfy, voros_val, 0.0)