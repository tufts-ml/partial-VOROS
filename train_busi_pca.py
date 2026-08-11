import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from metrics_jax import pvoros_score, pv_loss_fixed_thresh, soft_roc_fixed_thresholds
from metrics import pvoros_score as pvoros_score_np
import _geometry_jax
from pathlib import Path

DATA_DIR = Path("busi_training/busi_embeddings")       # expects DATA_DIR/{benign,malignant,normal}/*.png
RESULTS_DIR = Path(f"busi_training/results")
RESULTS_DIR.mkdir(exist_ok=True)
VAL_FRACTION = 0.2
SPLIT_SEED = 0

LR = 1e-4       # Lower learning rate for Partial VOROS
EPOCHS = 100


def split_train_val(feats, labels, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Split features and labels into training and validation sets"""
    return train_test_split(
        feats, labels,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )


def load_embeddings_and_labels(root: Path):
    """Load every embedding from the class directories under the BUSI embeddings root."""
    root = Path(root)
    label_map = {"benign": 0, "normal": 0, "malignant": 1}

    class_dirs = [root / name for name in ["benign", "malignant", "normal"] if (root / name).exists()]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories were found under {root}")

    embeddings = []
    labels = []

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


def convex_hull_roc(fprs, tprs):
    """Return the upper convex hull of an ROC curve."""
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
    """Plot iso-performance lines through each hull point using _geometry_jax._iso_performance_line."""
    if hull_pts.shape[0] == 0:
        return

    if fp_cost_ratios is None:
        fp_cost_ratios = np.linspace(0.05, 0.95, 5)

    x_vals = np.linspace(x_min, x_max, 200)

    for r in fp_cost_ratios:
        t = float(_geometry_jax.ratio_to_t(r, P, N))
        for h, k in hull_pts:
            a_j, b_j, c_j = _geometry_jax._iso_performance_line(h, k, t)
            a = float(a_j)
            b = float(b_j)
            c = float(c_j)

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


def init_params(key, dim):
    return {
        "theta": jax.random.normal(key, (dim,), dtype=jnp.float64) * 0.01,
        "c": jnp.array(0.0, dtype=jnp.float64),
    }


def train_baseline_logreg(feats, labels, seed=0):
    """Baseline (min-BCE) logistic regression classifier"""
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=int))
    return {
        "theta": np.asarray(clf.coef_[0], dtype=np.float64),
        "c": np.asarray(clf.intercept_[0], dtype=np.float64),
    }


def train_logreg(feats, labels, epochs=EPOCHS, lr=LR, seed=0, n_restarts=5, inits_per_seed=10):
    """Training procedure for soft PV-objective logistic regression """
    x = jnp.asarray(feats, dtype=jnp.float64)
    y = jnp.asarray(labels, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    # Constraints
    P = jnp.sum(y == 1.0)
    N = jnp.sum(y == 0.0)
    kappa = 0.5 * (P + N)
    alpha = 0.27
    min_fp_cost_ratio = 1 / 9
    max_fp_cost_ratio = 1 / 6

    
    # Loss
    def loss_fn(p):
        pv_params = {
            "w": p.get("theta", p.get("w")),
            "b": p.get("c", p.get("b")),
        }
        return pv_loss_fixed_thresh(
            pv_params, x, y, P, N, kappa, alpha,
            min_fp_cost_ratio, max_fp_cost_ratio
        )

    @jax.jit
    def train_step(params, opt_state):
        """Train step: update parameters based on gradient"""

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads



    def run_single_init(init_key):
        params = init_params(init_key, x.shape[1])
        opt_state = optimizer.init(params)

        best_params = params
        best_loss = float("inf")
        history = []

        for _ in range(epochs):
            params, opt_state, loss, _ = train_step(params, opt_state)
            curr_loss = float(loss)
            history.append(curr_loss)

            if curr_loss < best_loss and not np.isnan(curr_loss):
                best_loss = curr_loss
                best_params = params

        return best_params, best_loss, history

    best_overall_params = None
    best_overall_loss = float("inf")
    all_trace_histories = []

    # Run random initializations
    for i in range(n_restarts):
        run_seed = seed + i
        seed_key = jax.random.PRNGKey(run_seed)
        init_keys = jax.random.split(seed_key, inits_per_seed)

        print(f"\n--- Seed {run_seed} ({i + 1}/{n_restarts}) | Training {inits_per_seed} Weight Inits ---")
        
        seed_best_params = None
        seed_best_loss = float("inf")

        for init_idx, k in enumerate(init_keys):
            params, final_loss, history = run_single_init(k)

            # Save losses for plotting trace
            all_trace_histories.append({
                "seed": run_seed,
                "init_idx": init_idx,
                "history": history,
            })
            print(f"  └─ Init {init_idx + 1:2d}/{inits_per_seed} -> Loss: {final_loss:.4f}")

            if final_loss < seed_best_loss:
                seed_best_loss = final_loss
                seed_best_params = params

        print(f">> Best Loss for Seed {run_seed}: {seed_best_loss:.4f}")

        if seed_best_loss < best_overall_loss:
            best_overall_loss = seed_best_loss
            best_overall_params = seed_best_params

    print(f"\n[SUMMARY] Best Overall Loss across all seeds & inits: {best_overall_loss:.4f}")
    return best_overall_params, all_trace_histories


def plot_loss_traces(trace_histories, dataset_name, results_dir):
    """Plot soft PV loss over epoch for every initialization"""
    epochs_range = np.arange(1, EPOCHS + 1)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Identify best overall run to highlight
    best_final_loss = float("inf")
    best_history = None

    for run in trace_histories:
        hist = run["history"]
        if hist[-1] < best_final_loss:
            best_final_loss = hist[-1]
            best_history = hist

    # Plot all traces with light alpha
    for idx, run in enumerate(trace_histories):
        label = "Individual Inits" if idx == 0 else None
        ax.plot(epochs_range, run["history"], color='#1f77b4', alpha=0.25, lw=1.2, label=label)

    # Overlay best run
    if best_history is not None:
        ax.plot(epochs_range, best_history, color='#d62728', lw=2.5, label=f'Best Run (Min Loss = {best_final_loss:.4f})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Soft Partial VOROS Loss', fontsize=12)
    ax.set_title(f'Loss Trace across Initializations: {dataset_name}', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()

    filename_safe = dataset_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
    trace_plot_path = results_dir / f'loss_trace_{filename_safe}.pdf'
    fig.savefig(trace_plot_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved loss trace plot: {trace_plot_path}")


def train_logreg_from_baseline_init(feats, labels, init_params, epochs=EPOCHS, lr=LR, seed=0):
    """Fine-tune a trained logistic-regression initializer with the PV loss."""
    x = jnp.asarray(feats, dtype=jnp.float64)
    y = jnp.asarray(labels, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    P = jnp.sum(y == 1.0)
    N = jnp.sum(y == 0.0)
    kappa = 0.5 * (P + N)
    alpha = 0.27
    min_fp_cost_ratio = 1 / 9
    max_fp_cost_ratio = 1 / 6

    def loss_fn(p):
        pv_params = {
            "w": p.get("theta", p.get("w")),
            "b": p.get("c", p.get("b")),
        }
        return pv_loss_fixed_thresh(
            pv_params, x, y, P, N, kappa, alpha,
            min_fp_cost_ratio, max_fp_cost_ratio
        )

    params = {
        "theta": jnp.asarray(init_params["theta"], dtype=jnp.float64),
        "c": jnp.asarray(init_params["c"], dtype=jnp.float64),
    }
    opt_state = optimizer.init(params)

    best_params = params
    best_loss = float("inf")

    for epoch_idx in range(epochs):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        curr_loss = float(loss)

        if curr_loss < best_loss and not np.isnan(curr_loss):
            best_loss = curr_loss
            best_params = params

    print(f"[LR->PV init] seed={seed} | best loss={best_loss:.4f}")
    return best_params, best_loss


# ---------------------------------------------------------------------------
# 4. Main Pipeline (Loops across Full & PCA Reduced Dimensions)
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)

    X_train_raw, X_val_raw, y_train, y_val = split_train_val(all_feats, all_labels)

    print(f"Train samples: {X_train_raw.shape[0]}, Malignant rate: {y_train.mean():.3f}")
    print(f"Val samples:   {X_val_raw.shape[0]}, Malignant rate: {y_val.mean():.3f}")

    pca_dimensions = [2, 30, 120]
    
    datasets = {
        f"Full ({X_train_raw.shape[1]}D)": (X_train_raw, X_val_raw)
    }

    for dim in pca_dimensions:
        pca = PCA(n_components=dim, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        datasets[f"PCA {dim}D ({explained_var:.1f}% var)"] = (X_train_pca, X_val_pca)

    results_summary = {}

    for name, (X_train, X_val) in datasets.items():
        print("\n" + "=" * 65)
        print(f"        RUNNING EXPERIMENT: {name}")
        print("=" * 65)

        # Train model and extract loss histories for all initializations
        pv_params, trace_histories = train_logreg(X_train, y_train, n_restarts=1)
        
        # Plot and save trace curves for all initializations
        plot_loss_traces(trace_histories, name, RESULTS_DIR)

        baseline_params = train_baseline_logreg(X_train, y_train, seed=SPLIT_SEED)
        pv_from_baseline_params, pv_from_baseline_loss = train_logreg_from_baseline_init(
            X_train, y_train, baseline_params, epochs=EPOCHS, lr=LR, seed=SPLIT_SEED
        )

        x_val = jnp.asarray(X_val, dtype=jnp.float64)
        y_val_jax = jnp.asarray(y_val, dtype=jnp.float64)

        P_val = jnp.sum(y_val_jax == 1.0)
        N_val = jnp.sum(y_val_jax == 0.0)
        kappa_val = 0.5 * (P_val + N_val)
        alpha = 0.27
        min_fp_cost_ratio = 1 / 9
        max_fp_cost_ratio = 1 / 6
        n_points = 1000

        pv_logits_val = jnp.dot(x_val, pv_params["theta"]) + pv_params["c"]
        pv_y_pred_val = jax.nn.sigmoid(pv_logits_val)

        baseline_logits_val = jnp.asarray(
            np.dot(np.asarray(X_val, dtype=np.float64), baseline_params["theta"]) + baseline_params["c"],
            dtype=jnp.float64,
        )
        baseline_y_pred_val = jax.nn.sigmoid(baseline_logits_val)

        pv_from_baseline_logits_val = jnp.dot(x_val, pv_from_baseline_params["theta"]) + pv_from_baseline_params["c"]
        pv_from_baseline_y_pred_val = jax.nn.sigmoid(pv_from_baseline_logits_val)

        # Diagnostics
        print("\n--- VALIDATION PREDICTIONS DIAGNOSTIC ---")
        print(f"PV y_pred_val min : {float(jnp.min(pv_y_pred_val)):.6f}")
        print(f"PV y_pred_val max : {float(jnp.max(pv_y_pred_val)):.6f}")
        print(f"PV y_pred_val mean: {float(jnp.mean(pv_y_pred_val)):.6f}")
        print(f"PV y_pred_val std : {float(jnp.std(pv_y_pred_val)):.6f}")

        # Static Grid Score
        pv_fixed_thresh = pv_loss_fixed_thresh(
            pv_params, 
            x_val, 
            y_val_jax, 
            P_val, 
            N_val, 
            kappa_val, 
            alpha,
            min_fp_cost_ratio, 
            max_fp_cost_ratio, 
            n_points=n_points)
        pv_fixed_thresh = -float(pv_fixed_thresh)

        pv_from_baseline_fixed_thresh = pv_loss_fixed_thresh(
            pv_from_baseline_params,
            x_val,
            y_val_jax,
            P_val,
            N_val,
            kappa_val,
            alpha,
            min_fp_cost_ratio,
            max_fp_cost_ratio,
            n_points=n_points,
        )
        pv_from_baseline_fixed_thresh = -float(pv_from_baseline_fixed_thresh)

        # Empirical ROC VOROS Score
        fprs_emp, tprs_emp, _ = roc_curve(np.asarray(y_val), np.asarray(pv_y_pred_val))
        pv_score = pvoros_score(
            y_true=y_val,
            y_pred=pv_y_pred_val,
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points
        )
        pv_score_np = pvoros_score_np(
            y_true=np.asarray(y_val),
            y_pred=np.asarray(pv_y_pred_val),
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points
        )

        fprs_emp_from_baseline, tprs_emp_from_baseline, _ = roc_curve(
            np.asarray(y_val),
            np.asarray(pv_from_baseline_y_pred_val),
        )
        pv_from_baseline_score = pvoros_score(
            y_true=y_val,
            y_pred=pv_from_baseline_y_pred_val,
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points,
        )
        pv_from_baseline_score_np = pvoros_score_np(
            y_true=np.asarray(y_val),
            y_pred=np.asarray(pv_from_baseline_y_pred_val),
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points,
        )

        # Compute a smooth ROC curve alongside the discrete ROC for comparison
        soft_fprs, soft_tprs, _ = soft_roc_fixed_thresholds(
            y_val_jax,
            pv_y_pred_val,
            temp=0.02,
        )
        soft_fprs = np.asarray(soft_fprs, dtype=float)
        soft_tprs = np.asarray(soft_tprs, dtype=float)
        soft_fprs = np.clip(soft_fprs, 0.0, 1.0)
        soft_tprs = np.clip(soft_tprs, 0.0, 1.0)
        sort_idx = np.argsort(soft_fprs)
        soft_fprs = soft_fprs[sort_idx]
        soft_tprs = soft_tprs[sort_idx]

        pv_from_baseline_soft_fprs, pv_from_baseline_soft_tprs, _ = soft_roc_fixed_thresholds(
            y_val_jax,
            pv_from_baseline_y_pred_val,
            temp=0.02,
        )
        pv_from_baseline_soft_fprs = np.asarray(pv_from_baseline_soft_fprs, dtype=float)
        pv_from_baseline_soft_tprs = np.asarray(pv_from_baseline_soft_tprs, dtype=float)
        pv_from_baseline_soft_fprs = np.clip(pv_from_baseline_soft_fprs, 0.0, 1.0)
        pv_from_baseline_soft_tprs = np.clip(pv_from_baseline_soft_tprs, 0.0, 1.0)
        pv_from_baseline_soft_sort_idx = np.argsort(pv_from_baseline_soft_fprs)
        pv_from_baseline_soft_fprs = pv_from_baseline_soft_fprs[pv_from_baseline_soft_sort_idx]
        pv_from_baseline_soft_tprs = pv_from_baseline_soft_tprs[pv_from_baseline_soft_sort_idx]

        # Plot empirical ROC with alpha and kappa constraint bounds
        P = int(np.sum(y_val == 1))
        N = int(np.sum(y_val == 0))
        prevalence = P / (P + N)
        alpha_slope = alpha * (1 - prevalence) / (prevalence * (1 - alpha))
        kappa_slope = -(N / P)
        kappa_plot = 0.5 * (P + N)

        feasible_mask = np.array([
            _geometry_jax.keep_model(float(fpr), float(tpr), alpha, kappa_plot, N, P)
            for fpr, tpr in zip(fprs_emp, tprs_emp)
        ], dtype=bool)

        hull_pts = convex_hull_roc(
            fprs_emp[feasible_mask],
            tprs_emp[feasible_mask],
        )

        fpr_grid = np.linspace(0.0, 1.0, 200)
        tpr_alpha_bound = alpha_slope * fpr_grid
        tpr_kappa_bound = kappa_slope * fpr_grid + (kappa_plot / P)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fprs_emp, tprs_emp, color='#1f77b4', lw=2.5, label='PV ROC (discrete)')
        ax.plot(soft_fprs, soft_tprs, color='#ff7f0e', lw=2.0, label='PV ROC (smooth)')
        ax.plot(
            fprs_emp_from_baseline,
            tprs_emp_from_baseline,
            color='#9467bd',
            lw=2.0,
            linestyle='-.',
            label='LR-init PV ROC (discrete)',
        )
        ax.plot(
            pv_from_baseline_soft_fprs,
            pv_from_baseline_soft_tprs,
            color='#8c564b',
            lw=1.8,
            linestyle=':',
            label='LR-init PV ROC (smooth)',
        )
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance Baseline')
        ax.plot(fpr_grid, tpr_alpha_bound, color='#d62728', linestyle='--', lw=2.0, label=f'Alpha Bound (slope={alpha_slope:.2f})')
        ax.plot(fpr_grid, tpr_kappa_bound, color='#2ca02c', linestyle='--', lw=2.0, label=f'Kappa Bound (slope={kappa_slope:.2f})')

        fp_cost_ratios = np.linspace(min_fp_cost_ratio, max_fp_cost_ratio, 5)
        plot_iso_performance_lines(ax, hull_pts, P, N, fp_cost_ratios=fp_cost_ratios, x_min=0.0, x_max=1.0, color='black', alpha=0.12)

        y_lower_clamped = np.clip(tpr_alpha_bound, 0.0, 1.0)
        y_upper_clamped = np.clip(tpr_kappa_bound, 0.0, 1.0)
        ax.fill_between(
            fpr_grid,
            y_lower_clamped,
            y_upper_clamped,
            where=(y_upper_clamped >= y_lower_clamped),
            color='#ff7f0e', alpha=0.18, label='Feasible Region'
        )

        ax.scatter(hull_pts[:, 0], hull_pts[:, 1], color='black', s=30, zorder=5, label='ROC Hull Points')

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title(f'ROC with Alpha/Kappa Bounds: {name}', fontsize=13)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        fig.tight_layout()
        plot_name = RESULTS_DIR / f'roc_bounds_{name.replace(" ", "_").replace("/", "_")}.pdf'
        fig.savefig(plot_name, format='pdf', dpi=300)
        plt.close(fig)

        baseline_pv_score = pvoros_score(
            y_true=y_val,
            y_pred=baseline_y_pred_val,
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points
        )
        baseline_pv_score_np = pvoros_score_np(
            y_true=np.asarray(y_val),
            y_pred=np.asarray(baseline_y_pred_val),
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points
        )

        results_summary[name] = {
            "pv_fixed_thresh": pv_fixed_thresh * 100,
            "pv_score": float(pv_score) * 100,
            "pv_score_np": float(pv_score_np) * 100,
            "pv_from_baseline_fixed_thresh": pv_from_baseline_fixed_thresh * 100,
            "pv_from_baseline_score": float(pv_from_baseline_score) * 100,
            "pv_from_baseline_score_np": float(pv_from_baseline_score_np) * 100,
            "baseline_pv_score": float(baseline_pv_score) * 100,
            "baseline_pv_score_np": float(baseline_pv_score_np) * 100,
        }

        print(f"PV validation pVOROS (JAX): {pv_score:.4f}")
        print(f"PV validation pVOROS (NumPy): {pv_score_np:.4f}")
        print(f"LR-init PV validation pVOROS (JAX): {pv_from_baseline_score:.4f}")
        print(f"LR-init PV validation pVOROS (NumPy): {pv_from_baseline_score_np:.4f}")
        print(f"Baseline logistic regression pVOROS (JAX): {baseline_pv_score:.4f}")
        print(f"Baseline logistic regression pVOROS (NumPy): {baseline_pv_score_np:.4f}")

        # Cache final model weights per dimension
        dim_label = name.split()[1] if "PCA" in name else "full"
        np.save(RESULTS_DIR / f"logreg_theta_{dim_label}.npy", np.asarray(pv_params["theta"]))
        np.save(RESULTS_DIR / f"logreg_c_{dim_label}.npy", np.asarray(pv_params["c"]))
        np.save(RESULTS_DIR / f"baseline_logreg_theta_{dim_label}.npy", np.asarray(baseline_params["theta"]))
        np.save(RESULTS_DIR / f"baseline_logreg_c_{dim_label}.npy", np.asarray(baseline_params["c"]))
        np.save(RESULTS_DIR / f"lr_init_pv_theta_{dim_label}.npy", np.asarray(pv_from_baseline_params["theta"]))
        np.save(RESULTS_DIR / f"lr_init_pv_c_{dim_label}.npy", np.asarray(pv_from_baseline_params["c"]))

    # Summary
    print("\n" + "=" * 65)
    print("            FINAL DIMENSION COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Representation':<25} | {'Soft PV score (%)':<18} | {'PV score (%)':<18} | {'LR-init PV score (%)':<22} | {'Baseline PV score (%)':<22}")
    print("-" * 65)
    for name, metrics in results_summary.items():
        print(f"{name:<25} | {metrics['pv_fixed_thresh']:18.2f} | {metrics['pv_score']:18.2f} | {metrics['pv_from_baseline_score']:22.2f} | {metrics['baseline_pv_score']:22.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()