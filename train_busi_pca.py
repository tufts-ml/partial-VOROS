import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.decomposition import PCA
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

LR = 1e-3
EPOCHS = 100


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
        "w": jax.random.normal(key, (dim,), dtype=jnp.float64) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float64),
    }


def compute_pvoros_metric(params, feats_jax, labels_np, alpha=0.27, n_points=1000):
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


def bce_loss_fn(p, x, y):
    logits = jnp.dot(x, p["w"]) + p["b"]
    bce = jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    l2 = 1e-4 * jnp.sum(p["w"] ** 2)
    return bce + l2


# ---------------------------------------------------------------------------
# 2. Training Pipelines
# ---------------------------------------------------------------------------
def train_logreg_pv(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, seed=0, n_restarts=1, inits_per_seed=10):
    """Method 1: PV Loss from Random Initializations."""
    x_tr = jnp.asarray(X_train, dtype=jnp.float64)
    y_tr = jnp.asarray(y_train, dtype=jnp.float64)
    x_va = jnp.asarray(X_val, dtype=jnp.float64)
    y_va = jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    P_va, N_va = jnp.sum(y_va == 1.0), jnp.sum(y_va == 0.0)
    kappa_tr, kappa_va = 0.5 * (P_tr + N_tr), 0.5 * (P_va + N_va)
    alpha = 0.27
    min_fp, max_fp = 1 / 9, 1 / 6

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

        best_params = params
        best_val_pvoros = -float("inf")

        train_losses, val_losses = [], []
        train_pvoros_hist, val_pvoros_hist = [], []

        for ep in range(1, epochs + 1):
            params, opt_state, tr_loss = train_step(params, opt_state)
            va_loss = pv_loss_fixed_thresh(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp)

            train_losses.append(float(tr_loss))
            val_losses.append(float(va_loss))

            if ep % 10 == 0 or ep == 1:
                tr_pv = compute_pvoros_metric(params, x_tr, y_train)
                va_pv = compute_pvoros_metric(params, x_va, y_val)
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


def train_logreg_pv_from_bce_init(X_train, y_train, X_val, y_val, bce_init_params, epochs=EPOCHS, lr=LR):
    """Method 2: PV Loss starting from BCE Initializer."""
    x_tr = jnp.asarray(X_train, dtype=jnp.float64)
    y_tr = jnp.asarray(y_train, dtype=jnp.float64)
    x_va = jnp.asarray(X_val, dtype=jnp.float64)
    y_va = jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    P_va, N_va = jnp.sum(y_va == 1.0), jnp.sum(y_va == 0.0)
    kappa_tr, kappa_va = 0.5 * (P_tr + N_tr), 0.5 * (P_va + N_va)
    alpha = 0.27
    min_fp, max_fp = 1 / 9, 1 / 6

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)

    params = {
        "w": jnp.asarray(bce_init_params["w"], dtype=jnp.float64),
        "b": jnp.asarray(bce_init_params["b"], dtype=jnp.float64),
    }
    opt_state = optimizer.init(params)

    best_params = params
    best_val_pvoros = -float("inf")

    train_losses, val_losses = [], []
    train_pvoros_hist, val_pvoros_hist = [], []

    for ep in range(1, epochs + 1):
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        tr_loss = float(loss)
        va_loss = float(pv_loss_fixed_thresh(params, x_va, y_va, P_va, N_va, kappa_va, alpha, min_fp, max_fp))

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train)
            va_pv = compute_pvoros_metric(params, x_va, y_val)
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


def train_baseline_bce_methods(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=1e-2):
    """Methods 3 & 4: Trains BCE model and extracts BOTH checkpointing strategies.
    
    Returns:
      - best_bce_params: Method 3 (Standard BCE checkpointed on Best Val BCE Loss)
      - best_pvoros_params: Method 4 (BCE checkpointed on Best Val pVOROS Score)
      - history: Training curves and pVOROS traces
    """
    x_tr = jnp.asarray(X_train, dtype=jnp.float64)
    y_tr = jnp.asarray(y_train, dtype=jnp.float64)
    x_va = jnp.asarray(X_val, dtype=jnp.float64)
    y_va = jnp.asarray(y_val, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr)
    )

    key = jax.random.PRNGKey(SPLIT_SEED)
    params = init_params(key, x_tr.shape[1])
    opt_state = optimizer.init(params)

    best_bce_params = params
    best_val_bce_loss = float("inf")

    best_pvoros_params = params
    best_val_pvoros = -float("inf")

    train_bce_losses, val_bce_losses = [], []
    train_pvoros_hist, val_pvoros_hist = [], []

    for ep in range(1, epochs + 1):
        tr_bce_loss, grads = jax.value_and_grad(bce_loss_fn)(params, x_tr, y_tr)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        va_bce_loss = float(bce_loss_fn(params, x_va, y_va))
        train_bce_losses.append(float(tr_bce_loss))
        val_bce_losses.append(va_bce_loss)

        # Checkpoint Method 3: Best Validation BCE Loss
        if va_bce_loss < best_val_bce_loss:
            best_val_bce_loss = va_bce_loss
            best_bce_params = params

        # Evaluate and Checkpoint Method 4: Best Validation pVOROS Score every 10 epochs
        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train)
            va_pv = compute_pvoros_metric(params, x_va, y_val)
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
# 3. Plotting Traces
# ---------------------------------------------------------------------------
def plot_training_traces(pv_random_histories, pv_bce_init_history, bce_history, dataset_name, results_dir):
    """Plot Soft PV Loss (best random init + BCE init) and true pVOROS evaluation traces across epochs."""
    epochs_range = np.arange(1, EPOCHS + 1)
    filename_safe = dataset_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")

    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(16, 6))

    # Identify the single best random initialization run based on validation pVOROS performance
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
    # 1. Best Random Init PV Loss (Train vs. Val)
    ax_loss.plot(
        epochs_range, best_random_run["train_losses"],
        color='#1f77b4', lw=2.0, label='Best PV Random Init Train Loss'
    )
    ax_loss.plot(
        epochs_range, best_random_run["val_losses"],
        color='#1f77b4', linestyle='--', lw=2.0, label='Best PV Random Init Val Loss'
    )

    # 2. Soft PV Loss from BCE Initialization
    ax_loss.plot(
        epochs_range, pv_bce_init_history["train_losses"],
        color='#9467bd', lw=2.0, label='PV (BCE Init) Train Loss'
    )
    ax_loss.plot(
        epochs_range, pv_bce_init_history["val_losses"],
        color='#9467bd', linestyle='--', lw=2.0, label='PV (BCE Init) Val Loss'
    )

    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Soft Partial VOROS Loss', fontsize=12)
    ax_loss.set_title(f'Soft PV Loss Traces: {dataset_name}', fontsize=13)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------------
    # Right Panel: True Empirical pVOROS Score Traces (Every 10 Epochs)
    # ------------------------------------------------------------------
    # 1. Best PV Random Init
    tr_eps, tr_pvs = zip(*best_random_run["train_pvoros"])
    va_eps, va_pvs = zip(*best_random_run["val_pvoros"])
    ax_score.plot(tr_eps, tr_pvs, color='#2ca02c', marker='o', lw=2.0, label='PV Random Init Train pVOROS')
    ax_score.plot(va_eps, va_pvs, color='#2ca02c', marker='s', linestyle='--', lw=2.0, label='PV Random Init Val pVOROS')

    # 2. PV (BCE Init)
    bce_pv_tr_eps, bce_pv_tr_pvs = zip(*pv_bce_init_history["train_pvoros"])
    bce_pv_va_eps, bce_pv_va_pvs = zip(*pv_bce_init_history["val_pvoros"])
    ax_score.plot(bce_pv_tr_eps, bce_pv_tr_pvs, color='#8c564b', marker='^', lw=2.0, label='PV (BCE Init) Train pVOROS')
    ax_score.plot(bce_pv_va_eps, bce_pv_va_pvs, color='#8c564b', marker='v', linestyle='--', lw=2.0, label='PV (BCE Init) Val pVOROS')

    # 3. BCE Baseline (Monitored pVOROS trajectory)
    bce_tr_eps, bce_tr_pvs = zip(*bce_history["train_pvoros"])
    bce_va_eps, bce_va_pvs = zip(*bce_history["val_pvoros"])
    ax_score.plot(bce_tr_eps, bce_tr_pvs, color='#d62728', marker='D', lw=1.8, linestyle=':', label='BCE Baseline Train pVOROS')
    ax_score.plot(bce_va_eps, bce_va_pvs, color='#e377c2', marker='X', lw=1.8, linestyle=':', label='BCE Baseline Val pVOROS')

    ax_score.set_xlabel('Epoch', fontsize=12)
    ax_score.set_ylabel('True Empirical pVOROS Score', fontsize=12)
    ax_score.set_title(f'pVOROS Score Traces (Every 10 Epochs): {dataset_name}', fontsize=13)
    ax_score.grid(True, linestyle=':', alpha=0.6)
    ax_score.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    plot_path = results_dir / f'loss_pvoros_traces_{filename_safe}.pdf'
    fig.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved cleaned trace plot: {plot_path}")


# ---------------------------------------------------------------------------
# 4. Main Experiment Pipeline
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)

    # 60-20-20 Stratified Split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_val_test(all_feats, all_labels)

    print(f"Train samples: {X_train_raw.shape[0]} | Malignant rate: {y_train.mean():.3f}")
    print(f"Val samples:   {X_val_raw.shape[0]} | Malignant rate: {y_val.mean():.3f}")
    print(f"Test samples:  {X_test_raw.shape[0]} | Malignant rate: {y_test.mean():.3f}")

    pca_dimensions = [2, 30, 120]
    datasets = {
        f"Full ({X_train_raw.shape[1]}D)": (X_train_raw, X_val_raw, X_test_raw)
    }

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

        # Method 1: Soft PV Loss (Random Initializations)
        pv_rand_params, pv_rand_histories = train_logreg_pv(X_train, y_train, X_val, y_val, n_restarts=1)

        # Methods 3 & 4: Standard BCE Training (returns both BCE-loss checkpoint & PV-monitored checkpoint)
        bce_std_params, bce_monitored_params, bce_history = train_baseline_bce_methods(
            X_train, y_train, X_val, y_val, epochs=EPOCHS
        )

        # Method 2: Soft PV Loss (BCE Checkpoint Initializer)
        pv_bce_params, pv_bce_history = train_logreg_pv_from_bce_init(
            X_train, y_train, X_val, y_val, bce_std_params, epochs=EPOCHS, lr=LR
        )

        # Plot traces across all dynamic approaches
        plot_training_traces(pv_rand_histories, pv_bce_history, bce_history, name, RESULTS_DIR)

        # Convert Test inputs to JAX
        x_test_jax = jnp.asarray(X_test, dtype=jnp.float64)

        # Compute TEST pVOROS metrics for ALL 4 METHODS
        pv_rand_test_score = compute_pvoros_metric(pv_rand_params, x_test_jax, y_test)
        pv_bce_test_score = compute_pvoros_metric(pv_bce_params, x_test_jax, y_test)
        bce_std_test_score = compute_pvoros_metric(bce_std_params, x_test_jax, y_test)
        bce_monitored_test_score = compute_pvoros_metric(bce_monitored_params, x_test_jax, y_test)

        results_summary[name] = {
            "pv_rand_test": float(pv_rand_test_score) * 100,
            "pv_bce_test": float(pv_bce_test_score) * 100,
            "bce_std_test": float(bce_std_test_score) * 100,
            "bce_monitored_test": float(bce_monitored_test_score) * 100,
        }

        # Cache final model weights
        dim_label = name.split()[1] if "PCA" in name else "full"
        np.save(RESULTS_DIR / f"pv_rand_w_{dim_label}.npy", np.asarray(pv_rand_params["w"]))
        np.save(RESULTS_DIR / f"pv_rand_b_{dim_label}.npy", np.asarray(pv_rand_params["b"]))
        np.save(RESULTS_DIR / f"pv_bce_w_{dim_label}.npy", np.asarray(pv_bce_params["w"]))
        np.save(RESULTS_DIR / f"pv_bce_b_{dim_label}.npy", np.asarray(pv_bce_params["b"]))
        np.save(RESULTS_DIR / f"bce_std_w_{dim_label}.npy", np.asarray(bce_std_params["w"]))
        np.save(RESULTS_DIR / f"bce_std_b_{dim_label}.npy", np.asarray(bce_std_params["b"]))
        np.save(RESULTS_DIR / f"bce_monitored_w_{dim_label}.npy", np.asarray(bce_monitored_params["w"]))
        np.save(RESULTS_DIR / f"bce_monitored_b_{dim_label}.npy", np.asarray(bce_monitored_params["b"]))

    # Print Comparison Table
    print("\n" + "=" * 90)
    print("                     FINAL HELD-OUT TEST SET EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Representation':<22} | {'PV (Random Init)':<17} | {'PV (BCE Init)':<15} | {'BCE (Std Val BCE)':<18} | {'BCE (Monitored PV)':<18}")
    print("-" * 90)
    for name, metrics in results_summary.items():
        print(
            f"{name:<22} | "
            f"{metrics['pv_rand_test']:17.2f}% | "
            f"{metrics['pv_bce_test']:15.2f}% | "
            f"{metrics['bce_std_test']:18.2f}% | "
            f"{metrics['bce_monitored_test']:18.2f}%"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()