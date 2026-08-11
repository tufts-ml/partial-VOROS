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

DATA_DIR = Path("busi_training/busi_embeddings")
RESULTS_DIR = Path("busi_training/results")
RESULTS_DIR.mkdir(exist_ok=True)

VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 0

LR = 1e-4
EPOCHS = 100


# ---------------------------------------------------------------------------
# 1. Split & Data Loading
# ---------------------------------------------------------------------------
def split_train_val_test(feats, labels, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED):
    """Split features into 60% Train, 20% Val, and 20% Test sets (stratified)."""
    # First split off Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        feats, labels,
        test_size=test_frac,
        stratify=labels,
        random_state=seed,
    )
    # Next split remaining 80% into Train (60% total) and Val (20% total)
    relative_val_frac = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_frac,
        stratify=y_train_val,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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


# ---------------------------------------------------------------------------
# 2. Geometry & Plot Helpers
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


def init_params(key, dim):
    return {
        "w": jax.random.normal(key, (dim,), dtype=jnp.float64) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float64),
    }


def compute_pvoros_metric(params, feats_jax, labels_np, alpha=0.27, n_points=1000):
    """Evaluate true empirical pVOROS score using predictions."""
    logits = jnp.dot(feats_jax, params["w"]) + params["b"]
    y_pred = jax.nn.sigmoid(logits)
    score = pvoros_score(
        y_true=labels_np,
        y_pred=y_pred,
        alpha=alpha,
        kappa_frac=0.5,
        min_fp_cost_ratio=1/9,
        max_fp_cost_ratio=1/6,
        n_points=n_points
    )
    return float(score)


# ---------------------------------------------------------------------------
# 3. Training Procedures
# ---------------------------------------------------------------------------
def train_baseline_logreg(X_train, y_train):
    """Baseline standard BCE logistic regression classifier."""
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=SPLIT_SEED,
    )
    clf.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=int))
    return {
        "w": np.asarray(clf.coef_[0], dtype=np.float64),
        "b": np.asarray(clf.intercept_[0], dtype=np.float64),
    }


def train_logreg(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, seed=0, n_restarts=1, inits_per_seed=10):
    """Train PV-objective model, monitoring Train/Val Loss + True PVOROS every 10 epochs."""
    x_tr = jnp.asarray(X_train, dtype=jnp.float64)
    y_tr = jnp.asarray(y_train, dtype=jnp.float64)
    x_va = jnp.asarray(X_val, dtype=jnp.float64)
    y_va = jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    # Constraints
    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    kappa_tr = 0.5 * (P_tr + N_tr)
    P_va, N_va = jnp.sum(y_va == 1.0), jnp.sum(y_va == 0.0)
    kappa_va = 0.5 * (P_va + N_va)

    alpha = 0.27
    min_fp_cost_ratio, max_fp_cost_ratio = 1 / 9, 1 / 6

    def loss_fn(p, x, y, P, N, kappa):
        return pv_loss_fixed_thresh(
            p, x, y, P, N, kappa, alpha,
            min_fp_cost_ratio, max_fp_cost_ratio
        )

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_tr, y_tr, P_tr, N_tr, kappa_tr)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def run_single_init(init_key):
        params = init_params(init_key, x_tr.shape[1])
        opt_state = optimizer.init(params)

        best_params = params
        best_val_pvoros = -float("inf")

        train_losses, val_losses = [], []
        train_pvoros_hist, val_pvoros_hist = [], []

        for ep in range(1, epochs + 1):
            params, opt_state, tr_loss = train_step(params, opt_state)
            va_loss = loss_fn(params, x_va, y_va, P_va, N_va, kappa_va)

            train_losses.append(float(tr_loss))
            val_losses.append(float(va_loss))

            # Evaluate true empirical pVOROS score every 10 epochs
            if ep % 10 == 0 or ep == 1:
                tr_pv = compute_pvoros_metric(params, x_tr, y_train)
                va_pv = compute_pvoros_metric(params, x_va, y_val)
                train_pvoros_hist.append((ep, tr_pv))
                val_pvoros_hist.append((ep, va_pv))

                # Checkpoint parameters with the highest validation pVOROS score
                if va_pv > best_val_pvoros:
                    best_val_pvoros = va_pv
                    best_params = params

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
            all_trace_histories.append({
                "seed": run_seed,
                "init_idx": init_idx,
                "history": history,
            })
            print(f"  └─ Init {init_idx + 1:2d}/{inits_per_seed} -> Best Val pVOROS: {best_val_pv:.4f}")

            if best_val_pv > best_overall_val_pvoros:
                best_overall_val_pvoros = best_val_pv
                best_overall_params = params

    print(f"\n[SUMMARY] Best Overall Val pVOROS: {best_overall_val_pvoros:.4f}")
    return best_overall_params, all_trace_histories


def train_logreg_from_baseline_init(X_train, y_train, X_val, y_val, init_params, epochs=EPOCHS, lr=LR):
    """Fine-tune a baseline logistic regression initialization on PV loss with checkpointing."""
    x_tr = jnp.asarray(X_train, dtype=jnp.float64)
    y_tr = jnp.asarray(y_train, dtype=jnp.float64)
    x_va = jnp.asarray(X_val, dtype=jnp.float64)
    y_va = jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    kappa_tr = 0.5 * (P_tr + N_tr)
    alpha = 0.27
    min_fp_cost_ratio, max_fp_cost_ratio = 1 / 9, 1 / 6

    def loss_fn(p):
        return pv_loss_fixed_thresh(
            p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha,
            min_fp_cost_ratio, max_fp_cost_ratio
        )

    params = {
        "w": jnp.asarray(init_params["w"], dtype=jnp.float64),
        "b": jnp.asarray(init_params["b"], dtype=jnp.float64),
    }
    opt_state = optimizer.init(params)

    best_params = params
    best_val_pvoros = -float("inf")

    for ep in range(1, epochs + 1):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        if ep % 10 == 0 or ep == 1:
            va_pv = compute_pvoros_metric(params, x_va, y_val)
            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv
                best_params = params

    print(f"[LR->PV init] Best Checkpoint Val pVOROS: {best_val_pvoros:.4f}")
    return best_params


# ---------------------------------------------------------------------------
# 4. Plot Loss & Metric Trace Curves
# ---------------------------------------------------------------------------
def plot_training_traces(trace_histories, dataset_name, results_dir):
    """Plot Loss and true pVOROS evaluation traces over epochs."""
    epochs_range = np.arange(1, EPOCHS + 1)
    filename_safe = dataset_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")

    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15, 6))

    best_overall_val_pv = -float("inf")
    best_run_idx = 0

    # Locate best trajectory
    for idx, run in enumerate(trace_histories):
        if run["history"]["best_val_pvoros"] > best_overall_val_pv:
            best_overall_val_pv = run["history"]["best_val_pvoros"]
            best_run_idx = idx

    # Plot 1: Soft Loss (Train & Val)
    for idx, run in enumerate(trace_histories):
        is_best = (idx == best_run_idx)
        alpha = 0.8 if is_best else 0.2
        lw = 2.0 if is_best else 1.0

        ax_loss.plot(epochs_range, run["history"]["train_losses"], color='#1f77b4', alpha=alpha, lw=lw,
                     label='Train Loss' if is_best else None)
        ax_loss.plot(epochs_range, run["history"]["val_losses"], color='#ff7f0e', linestyle='--', alpha=alpha, lw=lw,
                     label='Val Loss' if is_best else None)

    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Soft Partial VOROS Loss', fontsize=12)
    ax_loss.set_title(f'Loss Traces: {dataset_name}', fontsize=13)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='upper right')

    # Plot 2: True pVOROS Score (evaluated every 10 epochs)
    for idx, run in enumerate(trace_histories):
        is_best = (idx == best_run_idx)
        alpha = 1.0 if is_best else 0.25
        lw = 2.5 if is_best else 1.0

        tr_eps, tr_pvs = zip(*run["history"]["train_pvoros"])
        va_eps, va_pvs = zip(*run["history"]["val_pvoros"])

        ax_score.plot(tr_eps, tr_pvs, color='#2ca02c', marker='o', alpha=alpha, lw=lw,
                      label='Train pVOROS' if is_best else None)
        ax_score.plot(va_eps, va_pvs, color='#d62728', marker='s', linestyle='--', alpha=alpha, lw=lw,
                      label='Val pVOROS' if is_best else None)

    ax_score.set_xlabel('Epoch', fontsize=12)
    ax_score.set_ylabel('True Empirical pVOROS Score', fontsize=12)
    ax_score.set_title(f'pVOROS Score Traces (Every 10 Epochs): {dataset_name}', fontsize=13)
    ax_score.grid(True, linestyle=':', alpha=0.6)
    ax_score.legend(loc='lower right')

    fig.tight_layout()
    plot_path = results_dir / f'loss_pvoros_traces_{filename_safe}.pdf'
    fig.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved trace plot: {plot_path}")


# ---------------------------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)

    # 60-20-20 Train / Val / Test Split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_val_test(all_feats, all_labels)

    print(f"Train samples: {X_train_raw.shape[0]} | Malignant rate: {y_train.mean():.3f}")
    print(f"Val samples:   {X_val_raw.shape[0]} | Malignant rate: {y_val.mean():.3f}")
    print(f"Test samples:  {X_test_raw.shape[0]} | Malignant rate: {y_test.mean():.3f}")

    pca_dimensions = [2, 30, 120]
    datasets = {
        f"Full ({X_train_raw.shape[1]}D)": (X_train_raw, X_val_raw, X_test_raw)
    }

    # Pre-compute PCA fitted strictly on Train
    for dim in pca_dimensions:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw)
        X_va = scaler.transform(X_val_raw)
        X_te = scaler.transform(X_test_raw)

        pca = PCA(n_components=dim, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)
        X_va_pca = pca.transform(X_va)
        X_te_pca = pca.transform(X_te)

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        datasets[f"PCA {dim}D ({explained_var:.1f}% var)"] = (X_tr_pca, X_va_pca, X_te_pca)

    results_summary = {}

    for name, (X_train, X_val, X_test) in datasets.items():
        print("\n" + "=" * 65)
        print(f"        RUNNING EXPERIMENT: {name}")
        print("=" * 65)

        # 1. Train JAX PV Model with monitoring & checkpointing
        pv_params, trace_histories = train_logreg(X_train, y_train, X_val, y_val, n_restarts=1)
        plot_training_traces(trace_histories, name, RESULTS_DIR)

        # 2. Train Standard BCE Baseline
        baseline_params = train_baseline_logreg(X_train, y_train)

        # 3. Fine-tune PV Model from BCE Initializer
        pv_from_baseline_params = train_logreg_from_baseline_init(
            X_train, y_train, X_val, y_val, baseline_params, epochs=EPOCHS, lr=LR
        )

        # Convert Test arrays to JAX
        x_test_jax = jnp.asarray(X_test, dtype=jnp.float64)

        # Predict probabilities on TEST set using best checkpointed models
        pv_test_preds = jax.nn.sigmoid(jnp.dot(x_test_jax, pv_params["w"]) + pv_params["b"])
        baseline_test_preds = jax.nn.sigmoid(np.dot(np.asarray(X_test, dtype=np.float64), baseline_params["w"]) + baseline_params["b"])
        pv_from_baseline_test_preds = jax.nn.sigmoid(jnp.dot(x_test_jax, pv_from_baseline_params["w"]) + pv_from_baseline_params["b"])

        # Compute TEST pVOROS metrics
        pv_test_score = compute_pvoros_metric(pv_params, x_test_jax, y_test)
        baseline_test_score = compute_pvoros_metric(baseline_params, x_test_jax, y_test)
        pv_from_baseline_test_score = compute_pvoros_metric(pv_from_baseline_params, x_test_jax, y_test)

        # Save summary
        results_summary[name] = {
            "pv_test_score": float(pv_test_score) * 100,
            "pv_from_baseline_test_score": float(pv_from_baseline_test_score) * 100,
            "baseline_test_score": float(baseline_test_score) * 100,
        }

        # Cache final checkpointed model weights
        dim_label = name.split()[1] if "PCA" in name else "full"
        np.save(RESULTS_DIR / f"logreg_w_{dim_label}.npy", np.asarray(pv_params["w"]))
        np.save(RESULTS_DIR / f"logreg_b_{dim_label}.npy", np.asarray(pv_params["b"]))
        np.save(RESULTS_DIR / f"baseline_logreg_w_{dim_label}.npy", np.asarray(baseline_params["w"]))
        np.save(RESULTS_DIR / f"baseline_logreg_b_{dim_label}.npy", np.asarray(baseline_params["b"]))

    # Summary
    print("\n" + "=" * 70)
    print("         FINAL TEST SET EVALUATION SUMMARY (BEST VAL CHECKPOINTS)")
    print("=" * 70)
    print(f"{'Representation':<25} | {'PV Test pVOROS (%)':<20} | {'LR-Init PV Test (%)':<20} | {'Baseline Test (%)':<18}")
    print("-" * 70)
    for name, metrics in results_summary.items():
        print(f"{name:<25} | {metrics['pv_test_score']:20.2f} | {metrics['pv_from_baseline_test_score']:20.2f} | {metrics['baseline_test_score']:18.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()