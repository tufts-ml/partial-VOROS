import os
import gc

# 1. Force JAX to allocate memory dynamically on demand rather than preallocating
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from metrics_jax import pvoros_score, pv_loss_fixed_thresh
from pathlib import Path
import argparse
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = Path("busi_training/busi_embeddings")
RESULTS_DIR = Path("busi_training/results")
RESULTS_DIR.mkdir(exist_ok=True)

GRIDSEARCH_DIR = RESULTS_DIR / "gs_a0_k0.5_1-9_1-6"
GRIDSEARCH_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 0
N_FOLDS = 5
EPOCHS = 100


# ---------------------------------------------------------------------------
# JIT-Compiled Loss Helper (Prevents Re-tracing & Memory Leaks in Loops)
# ---------------------------------------------------------------------------
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
    lr=1e-2, 
    seed=0, 
    n_restarts=1, 
    inits_per_seed=10, 
    wd=1e-2):
    """Method 1: Full-batch Soft PV Loss from Random Initializations."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float64), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=wd)
    )

    # Convert counts to static float scalars to prevent extra dynamic arrays
    P_tr, N_tr = float(np.sum(y_train == 1.0)), float(np.sum(y_train == 0.0))
    P_va, N_va = float(np.sum(y_val == 1.0)), float(np.sum(y_val == 0.0))
    kappa_tr, kappa_va = float(kappa_frac * (P_tr + N_tr)), float(kappa_frac * (P_va + N_va))

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
            
            # Use JITted evaluator instead of uncompiled raw function call
            va_loss = eval_pv_loss(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp)

            train_losses.append(float(tr_loss))
            val_losses.append(float(va_loss))

            if ep % 10 == 0 or ep == 1:
                tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
                va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
                train_pvoros_hist.append((ep, tr_pv))
                val_pvoros_hist.append((ep, va_pv))

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
            all_trace_histories.append({"seed": run_seed, "init_idx": init_idx, "history": history})
            print(f"  └─ Init {init_idx + 1:2d}/{inits_per_seed} -> Best Val pVOROS: {best_val_pv:.4f}")

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
    lr=1e-2, 
    wd=1e-2):
    """Method 2: Full-batch Soft PV Loss starting from BCE Initializer."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float32), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float32), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=wd)
    )

    P_tr, N_tr = float(np.sum(y_train == 1.0)), float(np.sum(y_train == 0.0))
    P_va, N_va = float(np.sum(y_val == 1.0)), float(np.sum(y_val == 0.0))
    kappa_tr, kappa_va = float(kappa_frac * (P_tr + N_tr)), float(kappa_frac * (P_va + N_va))

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state, loss

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
        params, opt_state, tr_loss = train_step(params, opt_state)
        
        # JITted loss evaluation
        va_loss = eval_pv_loss(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp)

        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))

        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
            va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
            train_pvoros_hist.append((ep, tr_pv))
            val_pvoros_hist.append((ep, va_pv))

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
    print(f"[PV Loss (BCE Init)] Best Checkpoint Val pVOROS: {best_val_pvoros:.4f}")
    return best_params, history


def train_baseline_bce_methods(X_train, y_train, X_val, y_val, alpha, kappa_frac, min_fp, max_fp, epochs=EPOCHS, lr=1e-2, wd=1e-2):
    """Methods 3 & 4: Full-batch BCE Training extracting both checkpointing strategies."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float64)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float64), jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=wd)
    )

    key = jax.random.PRNGKey(SPLIT_SEED)
    params = init_params(key, x_tr.shape[1])
    opt_state = optimizer.init(params)

    @jax.jit
    def bce_train_step(p, state):
        loss, grads = jax.value_and_grad(bce_loss_fn)(p, x_tr, y_tr)
        updates, state = optimizer.update(grads, state, params=p)
        return optax.apply_updates(p, updates), state, loss

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
        params, opt_state, tr_bce_loss = bce_train_step(params, opt_state)

        va_bce_loss = float(bce_loss_fn(params, x_va, y_va))
        train_bce_losses.append(float(tr_bce_loss))
        val_bce_losses.append(va_bce_loss)

        if va_bce_loss < best_val_bce_loss:
            best_val_bce_loss = va_bce_loss
            best_bce_params = params

        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train, alpha, kappa_frac, min_fp, max_fp)
            va_pv = compute_pvoros_metric(params, x_va, y_val, alpha, kappa_frac, min_fp, max_fp)
            train_pvoros_hist.append((ep, tr_pv))
            val_pvoros_hist.append((ep, va_pv))

            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv
                best_pvoros_params = params

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
# 3. Experiment Pipeline
# ---------------------------------------------------------------------------
def experiment():
    parser = argparse.ArgumentParser(description="5-Fold CV Grid Search Step")
    parser.add_argument("--w", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--s", type=float, default=1e-4, help="Step size / learning rate")
    args = parser.parse_args()

    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)
    X_train_raw, _, _, y_train, _, _ = split_train_val_test(all_feats, all_labels)

    pca_dimensions = [30]
    alpha, kappa_frac, min_fp, max_fp, epochs = 0, 0.5, 1 / 9, 1 / 6, 100

    val_records = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)

    for dim in pca_dimensions:
        name = f"Full ({X_train_raw.shape[1]}D)" if dim is None else f"PCA {dim}D"
        print("\n" + "=" * 65)
        print(f"        RUNNING 5-FOLD CV EXPERIMENT: {name}")
        print("=" * 65)

        fold_val_pv_rand = []
        fold_val_pv_bce = []
        fold_val_bce_loss = []
        fold_val_bce_monitored_pv = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train), start=1):
            X_tr, y_tr = X_train_raw[train_idx], y_train[train_idx]
            X_va, y_va = X_train_raw[val_idx], y_train[val_idx]

            # Fit transformers strictly on training fold
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

            if dim is not None:
                pca = PCA(n_components=dim, random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_va = pca.transform(X_va)

            # 1. Method 1: PV Random Init
            _, pv_hist = train_logreg_pv(
                X_tr, y_tr, X_va, y_va, alpha, kappa_frac, min_fp, max_fp, epochs=epochs, lr=args.s, wd=args.w
            )

            # 2 & 3. Methods 3 & 4: BCE Standard + BCE Monitored
            bce_std_params, _, bce_hist = train_baseline_bce_methods(
                X_tr, y_tr, X_va, y_va, alpha, kappa_frac, min_fp, max_fp, epochs=epochs, lr=args.s
            )

            # 4. Method 2: PV BCE Init
            _, pv_bce_hist = train_logreg_pv_from_bce_init(
                X_tr, y_tr, X_va, y_va, bce_std_params, alpha, kappa_frac, min_fp, max_fp, epochs=epochs, lr=args.s, wd=args.w
            )

            pv_rand_score = max(run["history"]["best_val_pvoros"] for run in pv_hist)

            fold_val_pv_rand.append(pv_rand_score)
            fold_val_pv_bce.append(pv_bce_hist['best_val_pvoros'])
            fold_val_bce_loss.append(-1 * bce_hist['best_val_bce_loss'])
            fold_val_bce_monitored_pv.append(bce_hist['best_val_pvoros'])

            print(f"  └─ Fold {fold}/{N_FOLDS} | Val PV (Rand): {pv_rand_score:.4f} | Val PV (BCE Init): {pv_bce_hist['best_val_pvoros']:.4f}")

            # Clear JAX cache & trigger garbage collection between folds
            jax.clear_caches()
            gc.collect()

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