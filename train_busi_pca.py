import argparse
import os
import numpy as np

# Default to CPU unless CUDA is explicitly available. This avoids the
# JAX backend initialization crash in environments without a GPU.
# os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from metrics_jax import pvoros_score, pv_loss_fixed_thresh, soft_roc_fixed_thresholds
from metrics import pvoros_score as pvoros_score_np
import _geometry_jax
from pathlib import Path

DATA_DIR = Path("/cluster/tufts/hugheslab/datasets/BUSI/ViT_embeddings")
RESULTS_DIR = Path("busi_training/results")
RESULTS_DIR.mkdir(exist_ok=True)

VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 0
CROSS_VAL_FOLDS = 5

LR = 1e-4
EPOCHS = 100
LR_CANDIDATES = [1e-5, 1e-4, 1e-3, 1e-2]
WD_CANDIDATES = [0, 1e-5, 1e-4, 1e-3, 1e-2]

@jax.jit
def eval_pv_loss(params, x, y, P, N, kappa, alpha, min_fp, max_fp):
    return pv_loss_fixed_thresh(params, x, y, P, N, kappa, alpha, min_fp, max_fp)

# ---------------------------------------------------------------------------
# 1. Data Splitting & Helper Procedures
# ---------------------------------------------------------------------------
def split_train_val_test(feats, labels, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED):
    """Split features into 60% Train, 20% Val, and 20% Test sets (stratified)."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        feats, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    relative_val_frac = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_frac, stratify=y_train_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_embeddings_and_labels(root: Path):
    root = Path(root)
    label_map = {"benign": 0, "normal": 0, "malignant": 1}

    class_dirs = [root / name for name in ["benign", "malignant", "normal"] if (root / name).exists()]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories were found under {root}")

    embeddings, labels = [], []
    for class_dir in class_dirs:
        class_name = class_dir.name
        for emb_path in sorted(class_dir.glob("*.pt")):
            emb = torch.load(emb_path, map_location="cpu")
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().reshape(-1).numpy()
            else:
                emb = np.asarray(emb).reshape(-1)
            embeddings.append(emb)
            labels.append(label_map[class_name])

    if not embeddings:
        raise ValueError(f"No embeddings were found in {root}")

    return np.vstack(embeddings), np.asarray(labels, dtype=int)


def init_params(key, dim):
    return {
        "w": jax.random.normal(key, (dim,), dtype=jnp.float32) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float32),
    }


def compute_pvoros_metric(params, feats_jax, labels_np, alpha, kappa_frac, min_fp, max_fp, n_points=1000):
    logits = jnp.dot(feats_jax, params["w"]) + params["b"]
    y_pred = jax.nn.sigmoid(logits)
    score = pvoros_score(
        y_true=labels_np,
        y_pred=y_pred,
        alpha=alpha,
        kappa_frac=kappa_frac,
        min_fp_cost_ratio=min_fp,
        max_fp_cost_ratio=max_fp,
        n_points=n_points
    )
    return float(score)


def bce_loss_fn(p, x, y):
    logits = jnp.dot(x, p["w"]) + p["b"]
    bce = jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    l2 = 1e-4 * jnp.sum(p["w"] ** 2)
    return bce + l2


# ---------------------------------------------------------------------------
# 2. Geometry & ROC Feasible Bound Plot Helpers
# ---------------------------------------------------------------------------
def convex_hull_roc(fprs, tprs):
    points = np.column_stack((fprs, tprs))
    hull = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - a[1]) - (a[1] - o[1]) * (b[0] - a[0])

    for pt in points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], pt) >= 0:
            hull.pop()
        hull.append(pt)

    return np.asarray(hull)


def plot_iso_performance_lines(ax, hull_pts, P, N, fp_cost_ratios=None, x_min=0.0, x_max=1.0, color='gray', alpha=0.2):
    if hull_pts.shape[0] == 0:
        return

    if fp_cost_ratios is None:
        fp_cost_ratios = np.linspace(0.05, 0.95, 5)

    x_vals = np.linspace(x_min, x_max, 200)

    for r in fp_cost_ratios:
        t = float(_geometry_jax.ratio_to_t(r, P, N))
        for h, k in hull_pts:
            a_j, b_j, c_j = _geometry_jax._iso_performance_line(h, k, t)
            a, b, c = float(a_j), float(b_j), float(c_j)

            if abs(b) < 1e-12:
                if abs(a) < 1e-12:
                    continue
                x_line = float(c / a)
                if x_line < x_min or x_line > x_max:
                    continue
                ax.plot([x_line, x_line], [0.0, 1.0], linestyle=':', color=color, alpha=alpha)
            else:
                y_vals = (c - a * x_vals) / b
                y_vals = np.clip(y_vals, 0.0, 1.0)
                ax.plot(x_vals, y_vals, linestyle=':', color=color, alpha=alpha)


def plot_roc_bounds_figure(y_val, y_pred_pv, y_pred_bce_monitored, y_pred_pv_bce_init, dataset_name, results_dir, alpha, kappa_frac, min_fp, max_fp):
    """Plots empirical ROC with alpha and kappa feasible bounds and iso-performance lines."""
    y_val_jax = jnp.asarray(y_val, dtype=jnp.float32)
    y_pred_pv_jax = jnp.asarray(y_pred_pv, dtype=jnp.float32)

    fprs_emp, tprs_emp, _ = roc_curve(y_val, y_pred_pv)
    fprs_emp_bce, tprs_emp_bce, _ = roc_curve(y_val, y_pred_bce_monitored)
    fprs_emp_pv_bce_init, tprs_emp_pv_bce_init, _ = roc_curve(y_val, y_pred_pv_bce_init)

    soft_fprs, soft_tprs, _ = soft_roc_fixed_thresholds(y_val_jax, y_pred_pv_jax, temp=0.02)
    soft_fprs = np.clip(np.asarray(soft_fprs, dtype=float), 0.0, 1.0)
    soft_tprs = np.clip(np.asarray(soft_tprs, dtype=float), 0.0, 1.0)
    sort_idx = np.argsort(soft_fprs)
    soft_fprs, soft_tprs = soft_fprs[sort_idx], soft_tprs[sort_idx]

    P = int(np.sum(y_val == 1))
    N = int(np.sum(y_val == 0))
    prevalence = P / (P + N)
    alpha_slope = alpha * (1 - prevalence) / (prevalence * (1 - alpha))
    kappa_slope = -(N / P)
    kappa_plot = kappa_frac * (P + N)

    feasible_mask = np.array([
        _geometry_jax.keep_model(float(fpr), float(tpr), alpha, kappa_plot, N, P)
        for fpr, tpr in zip(fprs_emp, tprs_emp)
    ], dtype=bool)

    hull_pts = convex_hull_roc(fprs_emp[feasible_mask], tprs_emp[feasible_mask])

    fpr_grid = np.linspace(0.0, 1.0, 200)
    tpr_alpha_bound = alpha_slope * fpr_grid
    tpr_kappa_bound = kappa_slope * fpr_grid + (kappa_plot / P)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fprs_emp, tprs_emp, color='#1f77b4', lw=2.5, label='PV ROC (discrete)')
    ax.plot(soft_fprs, soft_tprs, color='#ff7f0e', lw=2.0, label='PV ROC (smooth)')
    ax.plot(fprs_emp_bce, tprs_emp_bce, color='#9467bd', lw=2.0, linestyle='-.', label='BCE Monitored ROC')
    ax.plot(fprs_emp_pv_bce_init, tprs_emp_pv_bce_init, color="#51b9b9", lw=2.0, linestyle='-.', label='PV BCE init ROC')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance Baseline')
    ax.plot(fpr_grid, tpr_alpha_bound, color='#d62728', linestyle='--', lw=2.0, label=f'Alpha Bound (slope={alpha_slope:.2f})')
    ax.plot(fpr_grid, tpr_kappa_bound, color='#2ca02c', linestyle='--', lw=2.0, label=f'Kappa Bound (slope={kappa_slope:.2f})')

    fp_cost_ratios = np.linspace(min_fp, max_fp, 5)
    plot_iso_performance_lines(ax, hull_pts, P, N, fp_cost_ratios=fp_cost_ratios, x_min=0.0, x_max=1.0, color='black', alpha=0.12)

    y_lower_clamped = np.clip(tpr_alpha_bound, 0.0, 1.0)
    y_upper_clamped = np.clip(tpr_kappa_bound, 0.0, 1.0)
    ax.fill_between(
        fpr_grid, y_lower_clamped, y_upper_clamped,
        where=(y_upper_clamped >= y_lower_clamped),
        color='#ff7f0e', alpha=0.18, label='Feasible Region'
    )

    if hull_pts.shape[0] > 0:
        ax.scatter(hull_pts[:, 0], hull_pts[:, 1], color='black', s=30, zorder=5, label='ROC Hull Points')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC with Alpha/Kappa Bounds: {dataset_name}', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()

    filename_safe = dataset_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
    plot_name = results_dir / f'roc_bounds_{filename_safe}.pdf'
    fig.savefig(plot_name, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved ROC bounds plot: {plot_name}")


# ---------------------------------------------------------------------------
# 3. Full-Batch Optimization Pipelines
# ---------------------------------------------------------------------------
def train_logreg_pv(
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    alpha, 
    kappa_frac, 
    min_fp, 
    max_fp, 
    epochs=EPOCHS, 
    lr=LR, 
    seed=0, 
    n_restarts=1, 
    inits_per_seed=10, 
    weight_decay=1e-2):
    """Method 1: Full-batch Soft PV Loss from Random Initializations."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float32), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float32), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    P_va, N_va = jnp.sum(y_va == 1.0), jnp.sum(y_va == 0.0)
    kappa_tr, kappa_va = kappa_frac * (P_tr + N_tr), kappa_frac * (P_va + N_va)

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def run_single_init(init_key):
        params = init_params(init_key, x_tr.shape[1])
        opt_state = optimizer.init(params)

        

        train_losses, val_losses = [], []
        train_pvoros_hist, val_pvoros_hist = [], []

        tr_loss_0 = float(pure_loss_fn(params))
        va_loss_0 = float(eval_pv_loss(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp))
        train_losses.append(tr_loss_0)
        val_losses.append(va_loss_0)

        tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
        va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)   
        train_pvoros_hist.append((0, tr_pv))
        val_pvoros_hist.append((0, va_pv))
        best_params = params
        best_val_pvoros = va_pv

        for ep in range(1, epochs + 1):
            params, opt_state, tr_loss = train_step(params, opt_state)
            va_loss = pv_loss_fixed_thresh(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp)

            train_losses.append(float(tr_loss))
            val_losses.append(float(va_loss))

            # Compute true pVOROS score every 10 epochs
            if ep % 10 == 0 or ep == 1:
                tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
                va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
                train_pvoros_hist.append((ep, tr_pv))
                val_pvoros_hist.append((ep, va_pv))

                # Save best val PV checkpoint
                if va_pv > best_val_pvoros:
                    best_val_pvoros = va_pv
                    best_params = params
                # Print training state
                # print(f"[PV RAND] Epoch {ep:3d} | train_loss={float(tr_loss):.4f} | val_loss={float(va_loss):.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_pvoros": train_pvoros_hist,
            "val_pvoros": val_pvoros_hist,
            "best_val_pvoros": best_val_pvoros,
        }
        return best_params, best_val_pvoros, history

    best_overall_params = None
    best_overall_val_pvoros = -float("inf")
    all_trace_histories = []

    for i in range(n_restarts):
        run_seed = seed + i
        seed_key = jax.random.PRNGKey(run_seed)
        init_keys = jax.random.split(seed_key, inits_per_seed)

        print(f"\n--- Seed {run_seed} ({i + 1}/{n_restarts}) | Training {inits_per_seed} Weight Inits ---")

        for init_idx, k in enumerate(init_keys):
            params, best_val_pv, history = run_single_init(k)
            all_trace_histories.append({"seed": run_seed, "init_idx": init_idx, "history": history})
            print(f"  └─ Init {init_idx + 1:2d}/{inits_per_seed} -> Best Val pVOROS: {best_val_pv:.4f}")

            # Save best val PV checkpoint OVERALL
            if best_val_pv > best_overall_val_pvoros:
                best_overall_val_pvoros = best_val_pv
                best_overall_params = params

    print(f"\n[SUMMARY] Best Overall Val pVOROS: {best_overall_val_pvoros:.4f}")
    return best_overall_params, all_trace_histories


def train_logreg_pv_from_bce_init(
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    bce_init_params, 
    alpha, 
    kappa_frac, 
    min_fp, 
    max_fp, 
    epochs=EPOCHS, 
    lr=LR, 
    weight_decay=1e-2):
    """Method 2: Full-batch Soft PV Loss starting from BCE Initializer."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float32), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float32), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    P_va, N_va = jnp.sum(y_va == 1.0), jnp.sum(y_va == 0.0)
    kappa_tr, kappa_va = kappa_frac * (P_tr + N_tr), kappa_frac * (P_va + N_va)

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)


    # Initialize with BCE params
    params = {
        "w": jnp.asarray(bce_init_params["w"], dtype=jnp.float32),
        "b": jnp.asarray(bce_init_params["b"], dtype=jnp.float32),
    }
    opt_state = optimizer.init(params)

    

    train_losses, val_losses = [], []
    train_pvoros_hist, val_pvoros_hist = [], []

    tr_loss_0 = float(pure_loss_fn(params))
    va_loss_0 = float(eval_pv_loss(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp))
    train_losses.append(tr_loss_0)
    val_losses.append(va_loss_0)

    tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
    va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)   
    train_pvoros_hist.append((0, tr_pv))
    val_pvoros_hist.append((0, va_pv))
    best_params = params
    best_val_pvoros = va_pv

    for ep in range(1, epochs + 1):
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        tr_loss = float(loss)
        va_loss = float(pv_loss_fixed_thresh(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp))

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
            va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
            train_pvoros_hist.append((ep, tr_pv))
            val_pvoros_hist.append((ep, va_pv))


            # Save best val PV checkpoint
            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv
                best_params = params
            # Print training state
            # print(f"[PV BCE INIT] Epoch {ep:3d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_pvoros": train_pvoros_hist,
        "val_pvoros": val_pvoros_hist,
        "best_val_pvoros": best_val_pvoros,
    }
    print(f"[PV Loss (BCE Init)] Best Checkpoint Val pVOROS: {best_val_pvoros:.4f}")
    return best_params, history


def train_baseline_bce_methods(X_train, y_train, X_val, y_val, alpha, kappa_frac, min_fp, max_fp, epochs=EPOCHS, lr=1e-2, weight_decay=1e-2):
    """Methods 3 & 4: Full-batch BCE Training extracting both checkpointing strategies."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float32), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float32), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    )

    key = jax.random.PRNGKey(SPLIT_SEED)
    params = init_params(key, x_tr.shape[1])
    opt_state = optimizer.init(params)

    train_bce_losses, val_bce_losses = [], []
    train_pvoros_hist, val_pvoros_hist = [], []
    tr_bce_loss = float(bce_loss_fn(params, x_tr, y_tr))
    va_bce_loss = float(bce_loss_fn(params, x_va, y_va))
    train_bce_losses.append(tr_bce_loss)
    val_bce_losses.append(va_bce_loss)

    tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
    va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
    train_pvoros_hist.append((0, tr_pv))
    val_pvoros_hist.append((0, va_pv))

    best_bce_params = params
    best_val_bce_loss = va_bce_loss

    best_pvoros_params = params
    best_val_pvoros = va_pv

    for ep in range(1, epochs + 1):
        tr_bce_loss, grads = jax.value_and_grad(bce_loss_fn)(params, x_tr, y_tr)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        va_bce_loss = float(bce_loss_fn(params, x_va, y_va))
        train_bce_losses.append(float(tr_bce_loss))
        val_bce_losses.append(va_bce_loss)

        # Save params based on best val BCE 
        if va_bce_loss < best_val_bce_loss:
            best_val_bce_loss = va_bce_loss
            best_bce_params = params

        # Save params based on best val PV (every 10 epoch)
        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
            va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
            train_pvoros_hist.append((ep, tr_pv))
            val_pvoros_hist.append((ep, va_pv))

            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv
                best_pvoros_params = params
            # Print training state
            # print(f"[BCE BASE] Epoch {ep:3d} | train_bce={float(tr_bce_loss):.4f} | val_bce={va_bce_loss:.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

    history = {
        "train_bce_losses": train_bce_losses,
        "val_bce_losses": val_bce_losses,
        "train_pvoros": train_pvoros_hist,
        "val_pvoros": val_pvoros_hist,
        "best_val_bce_loss": best_val_bce_loss,
        "best_val_pvoros": best_val_pvoros,
    }
    print(f"[Baseline BCE (Standard)] Best Val BCE Loss: {best_val_bce_loss:.4f}")
    print(f"[Baseline BCE (Monitored PV)] Best Val pVOROS: {best_val_pvoros:.4f}")
    return best_bce_params, best_pvoros_params, history


# ---------------------------------------------------------------------------
# 4. Plot Training Traces
# ---------------------------------------------------------------------------
def plot_training_traces(pv_random_histories, pv_bce_init_history, bce_history, dataset_name, results_dir):
    """Plot Soft PV Loss (best random init + BCE init) and true pVOROS evaluation traces across epochs."""
    epochs_range = np.arange(0, EPOCHS + 1)
    filename_safe = dataset_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")

    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(16, 6))

    best_overall_val_pv = -float("inf")
    best_run_idx = 0
    for idx, run in enumerate(pv_random_histories):
        if run["history"]["best_val_pvoros"] > best_overall_val_pv:
            best_overall_val_pv = run["history"]["best_val_pvoros"]
            best_run_idx = idx

    best_random_run = pv_random_histories[best_run_idx]["history"]

    # ------------------------------------------------------------------
    # Left Panel: Soft Partial VOROS Loss Traces (Best Random Init + BCE Init)
    # ------------------------------------------------------------------
    ax_loss.plot(epochs_range, best_random_run["train_losses"], color='#1f77b4', lw=2.0, label='Best PV Random Init Train Loss')
    ax_loss.plot(epochs_range, best_random_run["val_losses"], color='#1f77b4', linestyle='--', lw=2.0, label='Best PV Random Init Val Loss')

    ax_loss.plot(epochs_range, pv_bce_init_history["train_losses"], color='#9467bd', lw=2.0, label='PV (BCE Init) Train Loss')
    ax_loss.plot(epochs_range, pv_bce_init_history["val_losses"], color='#9467bd', linestyle='--', lw=2.0, label='PV (BCE Init) Val Loss')

    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Soft Partial VOROS Loss', fontsize=12)
    ax_loss.set_title(f'Soft PV Loss Traces: {dataset_name}', fontsize=13)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------------
    # Right Panel: True Empirical pVOROS Score Traces (Every 10 Epochs)
    # ------------------------------------------------------------------
    tr_eps, tr_pvs = zip(*best_random_run["train_pvoros"])
    va_eps, va_pvs = zip(*best_random_run["val_pvoros"])
    ax_score.plot(tr_eps, tr_pvs, color='#2ca02c', marker='o', lw=2.0, label='PV Random Init Train pVOROS')
    ax_score.plot(va_eps, va_pvs, color='#2ca02c', marker='s', linestyle='--', lw=2.0, label='PV Random Init Val pVOROS')

    bce_pv_tr_eps, bce_pv_tr_pvs = zip(*pv_bce_init_history["train_pvoros"])
    bce_pv_va_eps, bce_pv_va_pvs = zip(*pv_bce_init_history["val_pvoros"])
    ax_score.plot(bce_pv_tr_eps, bce_pv_tr_pvs, color='#8c564b', marker='^', lw=2.0, label='PV (BCE Init) Train pVOROS')
    ax_score.plot(bce_pv_va_eps, bce_pv_va_pvs, color='#8c564b', marker='v', linestyle='--', lw=2.0, label='PV (BCE Init) Val pVOROS')

    bce_tr_eps, bce_tr_pvs = zip(*bce_history["train_pvoros"])
    bce_va_eps, bce_va_pvs = zip(*bce_history["val_pvoros"])
    ax_score.plot(bce_tr_eps, bce_tr_pvs, color='#d62728', marker='D', lw=1.8, linestyle=':', label='BCE (Monitored PV) Train pVOROS')
    ax_score.plot(bce_va_eps, bce_va_pvs, color='#e377c2', marker='X', lw=1.8, linestyle=':', label='BCE (Monitored PV) Val pVOROS')

    ax_bce = ax_score.twinx()
    ax_bce.plot(epochs_range, bce_history["train_bce_losses"], color='#ff9896', lw=1.5, alpha=0.85, label='BCE Train Loss')
    ax_bce.plot(epochs_range, bce_history["val_bce_losses"], color='#c49c94', lw=1.5, alpha=0.85, linestyle='--', label='BCE Val Loss')
    ax_bce.set_ylabel('BCE Loss', fontsize=12)

    ax_score.set_xlabel('Epoch', fontsize=12)
    ax_score.set_ylabel('True Empirical pVOROS Score', fontsize=12)
    ax_score.set_title(f'pVOROS & BCE Loss Traces: {dataset_name}', fontsize=13)
    ax_score.grid(True, linestyle=':', alpha=0.6)
    score_lines, score_labels = ax_score.get_legend_handles_labels()
    bce_lines, bce_labels = ax_bce.get_legend_handles_labels()
    ax_score.legend(score_lines + bce_lines, score_labels + bce_labels, loc='lower right', fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    plot_path = results_dir / f'loss_pvoros_traces_{filename_safe}.pdf'
    fig.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved trace plot: {plot_path}")


# ---------------------------------------------------------------------------
# 5. Main Experiment Pipeline
# ---------------------------------------------------------------------------
def tune_lr_with_cv(X, y, model_name, alpha, kappa_frac, min_fp, max_fp, target_dim=30, candidate_lrs=None, wd_candidates=None, n_splits=CROSS_VAL_FOLDS):
    """Tune learning rate for a model using 5-fold stratified CV on the train split."""
    if candidate_lrs is None:
        candidate_lrs = LR_CANDIDATES
    if wd_candidates is None:
        wd_candidates = WD_CANDIDATES

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SPLIT_SEED)
    grid_scores = {}
    best_lr = candidate_lrs[0]
    best_wd = wd_candidates[0]
    best_score = -float("inf")

    for lr in candidate_lrs:
        for wd in wd_candidates:
            fold_scores = []
            for train_idx, valid_idx in cv.split(X, y):
                X_tr, X_va = X[train_idx], X[valid_idx]
                y_tr, y_va = y[train_idx], y[valid_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)

                if target_dim is not None:
                    pca = PCA(n_components=target_dim, random_state=42)
                    X_tr = pca.fit_transform(X_tr)
                    X_va = pca.transform(X_va)

                if model_name == "pv_rand":
                    _, histories = train_logreg_pv(
                        X_tr, y_tr, X_va, y_va, alpha, kappa_frac, min_fp, max_fp,
                        epochs=EPOCHS, lr=lr, seed=SPLIT_SEED, n_restarts=1, inits_per_seed=10, weight_decay=wd
                    )
                    score = max(run["history"]["best_val_pvoros"] for run in histories)

                elif model_name == "pv_bce":
                    # FIX: Fit baseline BCE first to warm-start PV optimization cleanly
                    bce_std_params, _, _ = train_baseline_bce_methods(
                        X_tr, y_tr, X_va, y_va, alpha, kappa_frac, min_fp, max_fp, epochs=EPOCHS, lr=lr, weight_decay=wd
                    )
                    _, history = train_logreg_pv_from_bce_init(
                        X_tr, y_tr, X_va, y_va, bce_std_params,
                        alpha, kappa_frac, min_fp, max_fp, epochs=EPOCHS, lr=lr, weight_decay=wd
                    )
                    score = history["best_val_pvoros"]

                elif model_name == "bce":
                    _, _, history = train_baseline_bce_methods(
                        X_tr, y_tr, X_va, y_va, alpha, kappa_frac, min_fp, max_fp, epochs=EPOCHS, lr=lr, weight_decay=wd
                    )
                    score = history["best_val_pvoros"]
                else:
                    raise ValueError(f"Unknown model_name: {model_name}")

                fold_scores.append(float(score))

            mean_score = float(np.mean(fold_scores))
            grid_scores[(lr, wd)] = mean_score
            print(f"[CV Tune] {model_name:12s} | lr={lr:.1e} wd={wd:.1e} | mean val pVOROS={mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_lr = lr
                best_wd = wd

    return best_lr, best_wd, grid_scores


def _format_param_tag(value):
    """Filesystem-safe tag for a scalar hyperparameter."""
    from fractions import Fraction

    frac = Fraction(value).limit_denominator(1000)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}over{frac.denominator}"


def _config_label(alpha, kappa_frac, min_fp, max_fp):
    return (
        f"alpha{_format_param_tag(alpha)}"
        f"_kappa{_format_param_tag(kappa_frac)}"
        f"_minfp{_format_param_tag(min_fp)}"
        f"_maxfp{_format_param_tag(max_fp)}"
    )


FP_PAIRS = [
    (1 / 9, 1 / 6),
    (1, 10),
]

CONSTRAINT_CONFIGS = [
    {"alpha": 0.4, "kappa_frac": 1.0},
    {"alpha": 0.0, "kappa_frac": 0.5},
    {"alpha": 0.4, "kappa_frac": 0.5},
]

CONFIGS = [
    {"min_fp": min_fp, "max_fp": max_fp, **constraint}
    for min_fp, max_fp in FP_PAIRS
    for constraint in CONSTRAINT_CONFIGS
]


def main(pca_dimensions=None, lr_candidates=None, wd_candidates=None, cv_folds=CROSS_VAL_FOLDS, include_768=False, config_index=None):
    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)

    # 60-20-20 Stratified Split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_val_test(all_feats, all_labels)

    print(f"Train samples: {X_train_raw.shape[0]} | Malignant rate: {y_train.mean():.3f}")
    print(f"Val samples:   {X_val_raw.shape[0]} | Malignant rate: {y_val.mean():.3f}")
    print(f"Test samples:  {X_test_raw.shape[0]} | Malignant rate: {y_test.mean():.3f}")

    if pca_dimensions is None:
        pca_dimensions = [2, 30, 120]
    datasets = {}
    # Only include the full-dimension representation if it's not 768D or user explicitly requested 768D
    full_dim = X_train_raw.shape[1]
    if include_768 or full_dim != 768:
        datasets[f"Full ({full_dim}D)"] = ((X_train_raw, X_val_raw, X_test_raw), None)

    n_train, n_features = X_train_raw.shape[0], X_train_raw.shape[1]
    max_pca_rank = min(n_train, n_features)

    for dim in pca_dimensions:
        # Skip 768D PCA unless explicitly requested (avoid redundant full-dim run)
        if dim == 768 and not include_768:
            continue
        if dim > max_pca_rank:
            print(
                f"Skipping PCA {dim}D: n_components={dim} exceeds max rank "
                f"{max_pca_rank} (n_train={n_train}, n_features={n_features})"
            )
            continue
        # scaler = StandardScaler()
        # X_tr = scaler.fit_transform(X_train_raw)
        # X_va = scaler.transform(X_val_raw)
        # X_te = scaler.transform(X_test_raw)

        # pca = PCA(n_components=dim, random_state=42)
        # X_tr_pca = pca.fit_transform(X_tr)
        # X_va_pca = pca.transform(X_va)
        # X_te_pca = pca.transform(X_te)

        # explained_var = np.sum(pca.explained_variance_ratio_) * 100
        # datasets[f"PCA {dim}D ({explained_var:.1f}% var)"] = (X_tr_pca, X_va_pca, X_te_pca)
        datasets[f"PCA {dim}D"] = ((X_train_raw, X_val_raw, X_test_raw), dim)

    results_summary = {}

    if config_index is not None:
        configs_to_run = [CONFIGS[config_index]]
    else:
        configs_to_run = CONFIGS

    for constraint in configs_to_run:
            min_fp = constraint["min_fp"]
            max_fp = constraint["max_fp"]
            alpha = constraint["alpha"]
            kappa_frac = constraint["kappa_frac"]
            config_label = _config_label(alpha, kappa_frac, min_fp, max_fp)
            config_results_dir = RESULTS_DIR / config_label
            config_results_dir.mkdir(parents=True, exist_ok=True)
            results_summary[config_label] = {}

            print("\n" + "#" * 90)
            print(
                f"  CONSTRAINT CONFIG: alpha={alpha}, kappa_frac={kappa_frac}, "
                f"min_fp={min_fp}, max_fp={max_fp}"
            )
            print("#" * 90)

            for name, ((X_train, X_val, X_test), target_dim) in datasets.items():
                print("\n" + "=" * 65)
                print(f"        RUNNING EXPERIMENT: {name}")
                print(f"        Config: {config_label}")
                print("=" * 65)

                pv_rand_lr, pv_rand_wd, _ = tune_lr_with_cv(
                    X_train, y_train, "pv_rand", alpha, kappa_frac, min_fp, max_fp, target_dim=target_dim,
                    candidate_lrs=lr_candidates, wd_candidates=wd_candidates, n_splits=cv_folds
                )
                pv_bce_lr, pv_bce_wd, _ = tune_lr_with_cv(
                    X_train, y_train, "pv_bce", alpha, kappa_frac, min_fp, max_fp, target_dim=target_dim,
                    candidate_lrs=lr_candidates, wd_candidates=wd_candidates, n_splits=cv_folds
                )
                bce_lr, bce_wd, _ = tune_lr_with_cv(
                    X_train, y_train, "bce", alpha, kappa_frac, min_fp, max_fp, target_dim=target_dim,
                    candidate_lrs=lr_candidates, wd_candidates=wd_candidates, n_splits=cv_folds
                )

                print(f"[Selected LRs] PV random: {pv_rand_lr:.1e} | PV BCE init: {pv_bce_lr:.1e} | BCE: {bce_lr:.1e}")

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_va = scaler.transform(X_val)
                X_te = scaler.transform(X_test)

                if target_dim is not None:
                    pca = PCA(n_components=target_dim, random_state=42)
                    X_tr = pca.fit_transform(X_tr)
                    X_va = pca.transform(X_va)
                    X_te = pca.transform(X_te)

                # 1. Method 1: Soft PV Loss (Random Initializations)
                pv_rand_params, pv_rand_histories = train_logreg_pv(
                    X_tr, y_train, X_va, y_val, alpha, kappa_frac, min_fp, max_fp,
                    epochs=EPOCHS, lr=pv_rand_lr, n_restarts=1, weight_decay=pv_rand_wd
                )

                # 2. Methods 3 & 4: Standard BCE Training
                bce_std_params, bce_monitored_params, bce_history = train_baseline_bce_methods(
                    X_tr, y_train, X_va, y_val, alpha, kappa_frac, min_fp, max_fp,
                    epochs=EPOCHS, lr=bce_lr, weight_decay=bce_wd
                )

                # 3. Method 2: Soft PV Loss (BCE Checkpoint Initializer)
                pv_bce_params, pv_bce_history = train_logreg_pv_from_bce_init(
                    X_tr, y_train, X_va, y_val, bce_std_params, alpha, kappa_frac, min_fp, max_fp,
                    epochs=EPOCHS, lr=pv_bce_lr, weight_decay=pv_bce_wd
                )

                # 4. Generate Training Trace Plots
                plot_training_traces(pv_rand_histories, pv_bce_history, bce_history, name, config_results_dir)

                # 5. Generate ROC Curve Plot with Feasible Region and Iso-performance Lines
                x_val_jax = jnp.asarray(X_va, dtype=jnp.float32)
                pv_rand_val_preds = jax.nn.sigmoid(jnp.dot(x_val_jax, pv_rand_params["w"]) + pv_rand_params["b"])
                bce_monitored_val_preds = jax.nn.sigmoid(jnp.dot(x_val_jax, bce_monitored_params["w"]) + bce_monitored_params["b"])
                pv_bce_init_preds = jax.nn.sigmoid(jnp.dot(x_val_jax, pv_bce_params["w"]) + pv_bce_params["b"])

                plot_roc_bounds_figure(
                    y_val=y_val,
                    y_pred_pv=np.asarray(pv_rand_val_preds),
                    y_pred_bce_monitored=np.asarray(bce_monitored_val_preds),
                    y_pred_pv_bce_init=np.asarray(pv_bce_init_preds),
                    dataset_name=name,
                    results_dir=config_results_dir,
                    alpha=alpha, kappa_frac=kappa_frac, min_fp=min_fp, max_fp=max_fp
                )

                # 6. Evaluate TEST Set pVOROS metrics for ALL 4 METHODS
                x_test_jax = jnp.asarray(X_te, dtype=jnp.float32)
                pv_rand_test_score = compute_pvoros_metric(pv_rand_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)
                pv_bce_test_score = compute_pvoros_metric(pv_bce_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)
                bce_std_test_score = compute_pvoros_metric(bce_std_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)
                bce_monitored_test_score = compute_pvoros_metric(bce_monitored_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)

                results_summary[config_label][name] = {
                    "pv_rand_test": float(pv_rand_test_score) * 100,
                    "pv_bce_test": float(pv_bce_test_score) * 100,
                    "bce_std_test": float(bce_std_test_score) * 100,
                    "bce_monitored_test": float(bce_monitored_test_score) * 100,
                }

                # Cache model weights
                dim_label = name.split()[1] if "PCA" in name else "full"
                np.save(config_results_dir / f"pv_rand_w_{dim_label}.npy", np.asarray(pv_rand_params["w"]))
                np.save(config_results_dir / f"pv_rand_b_{dim_label}.npy", np.asarray(pv_rand_params["b"]))
                np.save(config_results_dir / f"pv_bce_w_{dim_label}.npy", np.asarray(pv_bce_params["w"]))
                np.save(config_results_dir / f"pv_bce_b_{dim_label}.npy", np.asarray(pv_bce_params["b"]))
                np.save(config_results_dir / f"bce_std_w_{dim_label}.npy", np.asarray(bce_std_params["w"]))
                np.save(config_results_dir / f"bce_std_b_{dim_label}.npy", np.asarray(bce_std_params["b"]))
                np.save(config_results_dir / f"bce_monitored_w_{dim_label}.npy", np.asarray(bce_monitored_params["w"]))
                np.save(config_results_dir / f"bce_monitored_b_{dim_label}.npy", np.asarray(bce_monitored_params["b"]))

                jax.clear_caches()
                gc.collect()

    # Summary Table
    print("\n" + "=" * 90)
    print("                     FINAL HELD-OUT TEST SET EVALUATION SUMMARY")
    print("=" * 90)
    for config_label, config_results in results_summary.items():
        print("\n" + "-" * 90)
        print(f"Config: {config_label}")
        print("-" * 90)
        print(f"{'Representation':<22} | {'PV (Random Init)':<17} | {'PV (BCE Init)':<15} | {'BCE (Std Val BCE)':<18} | {'BCE (Monitored PV)':<18}")
        print("-" * 90)
        for name, metrics in config_results.items():
            print(
                f"{name:<22} | "
                f"{metrics['pv_rand_test']:17.2f}% | "
                f"{metrics['pv_bce_test']:15.2f}% | "
                f"{metrics['bce_std_test']:18.2f}% | "
                f"{metrics['bce_monitored_test']:18.2f}%"
            )
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=str, default='2,30,120',
                        help='Comma-separated PCA dimensions to include (e.g. "2,30,120").')
    parser.add_argument('--lr_candidates', type=str, default=','.join(map(str, LR_CANDIDATES)),
                        help='Comma-separated learning rate candidates.')
    parser.add_argument('--wd_candidates', type=str, default=','.join(map(str, WD_CANDIDATES)),
                        help='Comma-separated weight-decay candidates.')
    parser.add_argument('--cv_folds', type=int, default=CROSS_VAL_FOLDS,
                        help='Number of CV folds to use when tuning.')
    parser.add_argument('--include_768', action='store_true', default=False,
                        help='Include the full raw 768D embedding representation (when available).')
    parser.add_argument('--config_index', type=int, default=None,
                        help='Run only CONFIGS[config_index] (0-based) instead of all constraint configs. '
                             'Useful for splitting configs across separate Slurm array jobs.')
    args = parser.parse_args()

    # Parse PCA dims
    if args.dims.strip() == '':
        pca_dims = []
    else:
        pca_dims = [int(x) for x in args.dims.split(',') if x.strip()]

    # Parse LR and WD candidate lists
    lr_candidates = [float(x) for x in args.lr_candidates.split(',') if x.strip()]
    wd_candidates = [float(x) for x in args.wd_candidates.split(',') if x.strip()]

    main(pca_dimensions=pca_dims, lr_candidates=lr_candidates, wd_candidates=wd_candidates, cv_folds=args.cv_folds, include_768=args.include_768, config_index=args.config_index)