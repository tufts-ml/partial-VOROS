import numpy as np
import os

# Prevent JAX memory issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from metrics_jax import pvoros_score, pv_loss_fixed_thresh
from pathlib import Path
import argparse
import pandas as pd

DATA_DIR = Path("busi_training/busi_embeddings")
RESULTS_DIR = Path("busi_training/results")
RESULTS_DIR.mkdir(exist_ok=True)

GRIDSEARCH_DIR = RESULTS_DIR / "gs_a0.4_k1.0_min1-9_max1-6"
GRIDSEARCH_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 0
N_FOLDS = 5


# ---------------------------------------------------------------------------
# 1. Data Splitting & Helper Procedures
# ---------------------------------------------------------------------------
def split_train_val_test(feats, labels, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED):
    """Reserves 20% held-out test data and returns 80% CV pool."""
    X_cv, X_test, y_cv, y_test = train_test_split(
        feats, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    return X_cv, X_test, y_cv, y_test


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
            emb = emb.detach().cpu().reshape(-1).numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb).reshape(-1)
            embeddings.append(emb)
            labels.append(label_map[class_name])

    if not embeddings:
        raise ValueError(f"No embeddings were found in {root}")

    return np.vstack(embeddings), np.asarray(labels, dtype=int)


def init_params(key, dim):
    return {
        "w": jax.random.normal(key, (dim,), dtype=jnp.float64) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float64),
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
# 2. Optimization Pipelines for 4 Methods
# ---------------------------------------------------------------------------
def train_logreg_pv(X_train, y_train, X_val, y_val, alpha, kappa_frac, min_fp, max_fp, epochs, lr, wd, seed=0, inits_per_seed=10):
    """Method 1: Soft PV Loss (Random Initialization)."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float64)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float64), jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=wd)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    kappa_tr = kappa_frac * (P_tr + N_tr)

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state, loss

    def run_single_init(init_key):
        params = init_params(init_key, x_tr.shape[1])
        opt_state = optimizer.init(params)
        best_val_pvoros = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)

        for ep in range(1, epochs + 1):
            params, opt_state, _ = train_step(params, opt_state)
            if ep % 10 == 0:
                va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
                if va_pv > best_val_pvoros:
                    best_val_pvoros = va_pv

        return best_val_pvoros

    seed_key = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(seed_key, inits_per_seed)
    best_overall_val_pvoros = max(run_single_init(k) for k in init_keys)
    return best_overall_val_pvoros


def train_logreg_pv_from_bce_init(X_train, y_train, X_val, y_val, bce_init_params, alpha, kappa_frac, min_fp, max_fp, epochs, lr, wd):
    """Method 2: Soft PV Loss starting from BCE Initializer."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float64)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float64), jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=wd)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    kappa_tr = kappa_frac * (P_tr + N_tr)

    def train_loss(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(train_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state, loss

    params = {
        "w": jnp.asarray(bce_init_params["w"], dtype=jnp.float64),
        "b": jnp.asarray(bce_init_params["b"], dtype=jnp.float64),
    }
    opt_state = optimizer.init(params)
    best_val_pvoros = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)

    for ep in range(1, epochs + 1):
        params, opt_state, _ = train_step(params, opt_state)
        if ep % 10 == 0:
            va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv

    return best_val_pvoros


def train_baseline_bce_methods(X_train, y_train, X_val, y_val, alpha, kappa_frac, min_fp, max_fp, epochs, lr):
    """Methods 3 & 4: Standard BCE and Monitored BCE Checkpoint."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float64)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float64), jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr)
    )

    key = jax.random.PRNGKey(SPLIT_SEED)
    params = init_params(key, x_tr.shape[1])
    opt_state = optimizer.init(params)

    best_bce_params = params
    best_val_bce_loss = float(bce_loss_fn(params, x_va, y_va))
    best_val_pvoros = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)

    for ep in range(1, epochs + 1):
        tr_bce_loss, grads = jax.value_and_grad(bce_loss_fn)(params, x_tr, y_tr)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        va_bce_loss = float(bce_loss_fn(params, x_va, y_va))
        if va_bce_loss < best_val_bce_loss:
            best_val_bce_loss = va_bce_loss
            best_bce_params = params

        if ep % 10 == 0:
            va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv

    return best_bce_params, best_val_bce_loss, best_val_pvoros


# ---------------------------------------------------------------------------
# 3. Experiment Pipeline
# ---------------------------------------------------------------------------
def experiment():
    parser = argparse.ArgumentParser(description="5-Fold CV Grid Search Step")
    parser.add_argument("--w", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--s", type=float, default=1e-4, help="Step size / learning rate")
    args = parser.parse_args()

    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)
    X_cv_raw, X_test_raw, y_cv, y_test = split_train_val_test(all_feats, all_labels)

    pca_dimensions = [None, 2, 30, 120]
    alpha, kappa_frac, min_fp, max_fp, epochs = 0.4, 1.0, 1 / 9, 1 / 6, 100

    val_records = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)

    for dim in pca_dimensions:
        name = f"Full ({X_cv_raw.shape[1]}D)" if dim is None else f"PCA {dim}D"
        print("\n" + "=" * 65)
        print(f"        RUNNING 5-FOLD CV EXPERIMENT: {name}")
        print("=" * 65)

        fold_val_pv_rand = []
        fold_val_pv_bce = []
        fold_val_bce_loss = []
        fold_val_bce_monitored_pv = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv_raw, y_cv), start=1):
            X_tr, y_train = X_cv_raw[train_idx], y_cv[train_idx]
            X_va, y_val = X_cv_raw[val_idx], y_cv[val_idx]

            # Fit transformers strictly on training fold
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

            if dim is not None:
                pca = PCA(n_components=dim, random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_va = pca.transform(X_va)

            # 1. Method 1: PV Random Init
            val_pv_rand = train_logreg_pv(
                X_tr, y_train, X_va, y_val, alpha, kappa_frac, min_fp, max_fp, epochs=epochs, lr=args.s, wd=args.w
            )

            # 2 & 3. Methods 3 & 4: BCE Standard + BCE Monitored
            bce_std_params, val_bce_loss, val_bce_monitored_pv = train_baseline_bce_methods(
                X_tr, y_train, X_va, y_val, alpha, kappa_frac, min_fp, max_fp, epochs=epochs, lr=args.s
            )

            # 4. Method 2: PV BCE Init
            val_pv_bce = train_logreg_pv_from_bce_init(
                X_tr, y_train, X_va, y_val, bce_std_params, alpha, kappa_frac, min_fp, max_fp, epochs=epochs, lr=args.s, wd=args.w
            )

            fold_val_pv_rand.append(val_pv_rand)
            fold_val_pv_bce.append(val_pv_bce)
            fold_val_bce_loss.append(-1 * val_bce_loss)
            fold_val_bce_monitored_pv.append(val_bce_monitored_pv)

            print(f"  └─ Fold {fold}/{N_FOLDS} | Val PV (Rand): {val_pv_rand:.4f} | Val PV (BCE Init): {val_pv_bce:.4f}")

        val_records.append({
            "dataset": name,
            "weight_decay": args.w,
            "learning_rate": args.s,
            "val_pv_rand_score": float(np.mean(fold_val_pv_rand)),
            "val_pv_bce_score": float(np.mean(fold_val_pv_bce)),
            "val_bce_score": float(np.mean(fold_val_bce_loss)),
            "val_bce_checkpoint_score": float(np.mean(fold_val_bce_monitored_pv)),
        })

    df_val = pd.DataFrame(val_records)
    out_csv = GRIDSEARCH_DIR / f"cv_val_results_w{args.w}_s{args.s}.csv"
    df_val.to_csv(out_csv, index=False)
    print(f"\n[SUMMARY] Successfully logged 5-Fold CV Scores: {out_csv}", flush=True)


if __name__ == "__main__":
    experiment()