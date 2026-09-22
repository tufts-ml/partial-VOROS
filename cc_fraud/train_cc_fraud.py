import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
import optax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

from metrics_jax import pvoros_score, pv_loss_fixed_thresh

DATA_DIR = "/cluster/home/jli48/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VAL_FRACTION = 1 / 6
TEST_FRACTION = 1 / 6
SPLIT_SEED = 0

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50

ALPHA = 0.1
KAPPA_FRAC = 0.5
MIN_FP = 1 / 9
MAX_FP = 1 / 6


# ---------------------------------------------------------------------------
# 1. Data loading & splitting
# ---------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(Path(csv_path) / "creditcard.csv")
    labels = df["Class"].to_numpy(dtype=int)
    feats = df.drop(columns=["Class"]).to_numpy(dtype=np.float64)
    return feats, labels


def split_train_val_test(feats, labels, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED):
    """Split features into 4:1:1 Train, Val, Test sets (stratified)."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        feats, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    relative_val_frac = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_frac, stratify=y_train_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def init_params(key, dim):
    return {
        "w": jax.random.normal(key, (dim,), dtype=jnp.float32) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float32),
    }


def bce_loss_fn(p, x, y):
    logits = jnp.dot(x, p["w"]) + p["b"]
    bce = jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    l2 = 1e-4 * jnp.sum(p["w"] ** 2)
    return bce + l2


def compute_pvoros_metric(params, feats_jax, labels_np, alpha=ALPHA, kappa_frac=KAPPA_FRAC, min_fp=MIN_FP, max_fp=MAX_FP, n_points=1000):
    logits = jnp.dot(feats_jax, params["w"]) + params["b"]
    y_pred = jax.nn.sigmoid(logits)
    score = pvoros_score(
        y_true=labels_np,
        y_pred=y_pred,
        alpha=alpha,
        kappa_frac=kappa_frac,
        min_fp_cost_ratio=min_fp,
        max_fp_cost_ratio=max_fp,
        n_points=n_points,
    )
    return float(score)


def make_optimizer(lr=LR, weight_decay=WEIGHT_DECAY):
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    )


# ---------------------------------------------------------------------------
# 2. Training loops
# ---------------------------------------------------------------------------
def train_bce_with_checkpoints(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, seed=SPLIT_SEED):
    """Single full-batch BCE training run, tracking two checkpoints:
    best val BCE loss (Model 1) and best train pVOROS (Model 5)."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float32), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float32), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = make_optimizer(lr, weight_decay)
    key = jax.random.PRNGKey(seed)
    params = init_params(key, x_tr.shape[1])
    opt_state = optimizer.init(params)

    best_val_bce_params = params
    best_val_bce_loss = float("inf")

    best_train_pvoros_params = params
    best_train_pvoros = -float("inf")

    train_bce_losses, val_bce_losses = [], []
    train_pvoros_hist, val_pvoros_hist = [], []

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(bce_loss_fn)(params, x_tr, y_tr)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for ep in range(1, epochs + 1):
        params, opt_state, tr_loss = train_step(params, opt_state)
        va_loss = float(bce_loss_fn(params, x_va, y_va))

        train_bce_losses.append(float(tr_loss))
        val_bce_losses.append(va_loss)

        if va_loss < best_val_bce_loss:
            best_val_bce_loss = va_loss
            best_val_bce_params = params

        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train)
            va_pv = compute_pvoros_metric(params, x_va, y_val)
            train_pvoros_hist.append((ep, tr_pv))
            val_pvoros_hist.append((ep, va_pv))

            if tr_pv > best_train_pvoros:
                best_train_pvoros = tr_pv
                best_train_pvoros_params = params

            print(f"[BCE] Epoch {ep:3d} | train_bce={float(tr_loss):.4f} | val_bce={va_loss:.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

    history = {
        "train_losses": train_bce_losses,
        "val_losses": val_bce_losses,
        "train_pvoros": train_pvoros_hist,
        "val_pvoros": val_pvoros_hist,
    }
    print(f"[BCE Baseline] Best Val BCE Loss: {best_val_bce_loss:.4f}")
    print(f"[BCE Track-Train-PVOROS] Best Train pVOROS: {best_train_pvoros:.4f}")
    return best_val_bce_params, best_train_pvoros_params, history


def train_pvoros_loss(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, seed=SPLIT_SEED, init_from=None):
    """Full-batch soft PV-loss training. If init_from is None, random-inits
    (Model 3); otherwise starts from the given params (Model 4)."""
    x_tr, y_tr = jnp.asarray(X_train, dtype=jnp.float32), jnp.asarray(y_train, dtype=jnp.float32)
    x_va, y_va = jnp.asarray(X_val, dtype=jnp.float32), jnp.asarray(y_val, dtype=jnp.float32)

    optimizer = make_optimizer(lr, weight_decay)

    if init_from is None:
        key = jax.random.PRNGKey(seed)
        params = init_params(key, x_tr.shape[1])
    else:
        params = {
            "w": jnp.asarray(init_from["w"], dtype=jnp.float32),
            "b": jnp.asarray(init_from["b"], dtype=jnp.float32),
        }
    opt_state = optimizer.init(params)

    P_tr, N_tr = jnp.sum(y_tr == 1.0), jnp.sum(y_tr == 0.0)
    kappa_tr = KAPPA_FRAC * (P_tr + N_tr)

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr, P_tr, N_tr, kappa_tr, ALPHA, MIN_FP, MAX_FP)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    best_params = params
    best_val_pvoros = -float("inf")

    train_losses, val_losses = [], []
    train_pvoros_hist, val_pvoros_hist = [], []

    for ep in range(1, epochs + 1):
        params, opt_state, tr_loss = train_step(params, opt_state)
        y_va_1d = y_va
        P_va, N_va = jnp.sum(y_va_1d == 1.0), jnp.sum(y_va_1d == 0.0)
        kappa_va = KAPPA_FRAC * (P_va + N_va)
        va_loss = float(pv_loss_fixed_thresh(params, x_va, y_va, P_va, N_va, kappa_va, ALPHA, MIN_FP, MAX_FP))

        train_losses.append(float(tr_loss))
        val_losses.append(va_loss)

        if ep % 10 == 0 or ep == 1:
            tr_pv = compute_pvoros_metric(params, x_tr, y_train)
            va_pv = compute_pvoros_metric(params, x_va, y_val)
            train_pvoros_hist.append((ep, tr_pv))
            val_pvoros_hist.append((ep, va_pv))

            if va_pv > best_val_pvoros:
                best_val_pvoros = va_pv
                best_params = params

            print(f"[PV] Epoch {ep:3d} | train_loss={float(tr_loss):.4f} | val_loss={va_loss:.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_pvoros": train_pvoros_hist,
        "val_pvoros": val_pvoros_hist,
    }
    print(f"[PV Loss] Best Val pVOROS: {best_val_pvoros:.4f}")
    return best_params, history


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def plot_model_traces(history, model_name, results_dir, loss_label="Loss"):
    epochs_range = np.arange(1, len(history["train_losses"]) + 1)

    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax_loss.plot(epochs_range, history["train_losses"], color='#1f77b4', lw=2.0, label=f'Train {loss_label}')
    ax_loss.plot(epochs_range, history["val_losses"], color='#1f77b4', linestyle='--', lw=2.0, label=f'Val {loss_label}')
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel(loss_label, fontsize=12)
    ax_loss.set_title(f'{loss_label} Traces: {model_name}', fontsize=13)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='best', fontsize=9, framealpha=0.9)

    tr_eps, tr_pvs = zip(*history["train_pvoros"])
    va_eps, va_pvs = zip(*history["val_pvoros"])
    ax_score.plot(tr_eps, tr_pvs, color='#2ca02c', marker='o', lw=2.0, label='Train pVOROS')
    ax_score.plot(va_eps, va_pvs, color='#d62728', marker='s', linestyle='--', lw=2.0, label='Val pVOROS')
    ax_score.set_xlabel('Epoch', fontsize=12)
    ax_score.set_ylabel('pVOROS Score', fontsize=12)
    ax_score.set_title(f'pVOROS Traces: {model_name}', fontsize=13)
    ax_score.grid(True, linestyle=':', alpha=0.6)
    ax_score.legend(loc='best', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    plot_path = results_dir / f'{model_name}_traces.pdf'
    fig.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved trace plot: {plot_path}")


def plot_model_roc(y_test, y_pred, model_name, results_dir):
    fprs, tprs, _ = roc_curve(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(fprs, tprs, color='#1f77b4', lw=2.5, label=f'ROC (AUROC={auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance Baseline')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'Test ROC: {model_name}', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()

    plot_path = results_dir / f'{model_name}_roc.pdf'
    fig.savefig(plot_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved ROC plot: {plot_path}")

    return auroc


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = load_data(DATA_DIR)
    print(f"Loaded {all_feats.shape[0]} rows, {all_feats.shape[1]} features, "
          f"fraud rate={all_labels.mean():.5f}")

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_val_test(all_feats, all_labels)
    print(f"Train samples: {X_train_raw.shape[0]} | Fraud rate: {y_train.mean():.5f}")
    print(f"Val samples:   {X_val_raw.shape[0]} | Fraud rate: {y_val.mean():.5f}")
    print(f"Test samples:  {X_test_raw.shape[0]} | Fraud rate: {y_test.mean():.5f}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    x_test_jax = jnp.asarray(X_test, dtype=jnp.float32)

    results = {}
    model_params = {}

    # Models 1 & 5: shared BCE run, two checkpoints.
    print("\n" + "=" * 70)
    print("  Training BCE run (Models 1 & 5)")
    print("=" * 70)
    bce_baseline_params, bce_track_train_pvoros_params, bce_history = train_bce_with_checkpoints(
        X_train, y_train, X_val, y_val
    )
    model_params["bce_baseline"] = bce_baseline_params
    model_params["bce_track_train_pvoros"] = bce_track_train_pvoros_params
    plot_model_traces(bce_history, "bce_baseline", RESULTS_DIR, loss_label="BCE Loss")
    plot_model_traces(bce_history, "bce_track_train_pvoros", RESULTS_DIR, loss_label="BCE Loss")

    # Model 3: pVOROS trained + validated, random init.
    print("\n" + "=" * 70)
    print("  Training pVOROS run (Model 3, random init)")
    print("=" * 70)
    pvoros_params, pvoros_history = train_pvoros_loss(X_train, y_train, X_val, y_val, init_from=None)
    model_params["pvoros"] = pvoros_params
    plot_model_traces(pvoros_history, "pvoros", RESULTS_DIR, loss_label="Soft PV Loss")

    # Model 4: pVOROS trained + validated, init'd from Model 1's weights.
    print("\n" + "=" * 70)
    print("  Training pVOROS run (Model 4, init from Model 1)")
    print("=" * 70)
    pvoros_from_bce_init_params, pvoros_from_bce_init_history = train_pvoros_loss(
        X_train, y_train, X_val, y_val, init_from=bce_baseline_params
    )
    model_params["pvoros_from_bce_init"] = pvoros_from_bce_init_params
    plot_model_traces(pvoros_from_bce_init_history, "pvoros_from_bce_init", RESULTS_DIR, loss_label="Soft PV Loss")

    # Evaluate all models on test set.
    rows = []
    for model_name, params in model_params.items():
        test_pvoros = compute_pvoros_metric(params, x_test_jax, y_test) * 100
        y_test_pred = np.asarray(jax.nn.sigmoid(jnp.dot(x_test_jax, params["w"]) + params["b"]))
        test_auroc = plot_model_roc(y_test, y_test_pred, model_name, RESULTS_DIR)

        rows.append({"model_name": model_name, "test_pvoros_pct": test_pvoros, "test_auroc": test_auroc})

        np.save(RESULTS_DIR / f"{model_name}_w.npy", np.asarray(params["w"]))
        np.save(RESULTS_DIR / f"{model_name}_b.npy", np.asarray(params["b"]))

    results_df = pd.DataFrame(rows)
    results_csv_path = RESULTS_DIR / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST SET EVALUATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved results table: {results_csv_path}")


if __name__ == "__main__":
    main()
