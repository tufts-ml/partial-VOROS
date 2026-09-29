import argparse
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
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
WEIGHT_DECAY = 1e-4
EPOCHS = 100
N_INITS = 10

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


def compute_pvoros_metric(params, feats_jax, labels_np, alpha=None, kappa_frac=None, min_fp=None, max_fp=None, n_points=1000):
    # Defaults are resolved from the module-level globals at call time (not
    # bind time) so that overriding ALPHA/KAPPA_FRAC/MIN_FP/MAX_FP from CLI
    # args in main() takes effect here too.
    alpha = ALPHA if alpha is None else alpha
    kappa_frac = KAPPA_FRAC if kappa_frac is None else kappa_frac
    min_fp = MIN_FP if min_fp is None else min_fp
    max_fp = MAX_FP if max_fp is None else max_fp
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
def train_bce_with_checkpoints(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, seed=SPLIT_SEED, verbose=True):
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

            if verbose:
                print(f"[BCE] Epoch {ep:3d} | train_bce={float(tr_loss):.4f} | val_bce={va_loss:.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

    history = {
        "train_losses": train_bce_losses,
        "val_losses": val_bce_losses,
        "train_pvoros": train_pvoros_hist,
        "val_pvoros": val_pvoros_hist,
    }
    if verbose:
        print(f"[BCE Baseline] Best Val BCE Loss: {best_val_bce_loss:.4f}")
        print(f"[BCE Track-Train-PVOROS] Best Train pVOROS: {best_train_pvoros:.4f}")
    return best_val_bce_params, best_val_bce_loss, best_train_pvoros_params, best_train_pvoros, history


def train_pvoros_loss(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, seed=SPLIT_SEED, init_from=None, verbose=True):
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

            if verbose:
                print(f"[PV] Epoch {ep:3d} | train_loss={float(tr_loss):.4f} | val_loss={va_loss:.4f} | train_pv={tr_pv:.4f} | val_pv={va_pv:.4f}")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_pvoros": train_pvoros_hist,
        "val_pvoros": val_pvoros_hist,
    }
    if verbose:
        print(f"[PV Loss] Best Val pVOROS: {best_val_pvoros:.4f}")
    return best_params, best_val_pvoros, history


# ---------------------------------------------------------------------------
# 3. Learning-rate tuning (grid over LR_GRID x N_INITS random restarts)
# ---------------------------------------------------------------------------
def tune_and_train_bce(X_train, y_train, X_val, y_val, lr_grid=LR_GRID, n_inits=N_INITS, seed=SPLIT_SEED):
    """Grid-search LR for the shared BCE run. The winning LR is the one whose
    best-of-N-inits val BCE loss is lowest (bce_baseline's metric). Both
    bce_baseline and bce_track_train_pvoros are then read off runs at that
    single winning LR, since they come from the same physical training run."""
    per_lr_runs = {}
    for lr in lr_grid:
        runs = []
        for init_idx in range(n_inits):
            init_seed = seed + init_idx
            bb_params, val_loss, tp_params, train_pvoros, history = train_bce_with_checkpoints(
                X_train, y_train, X_val, y_val, lr=lr, seed=init_seed, verbose=False
            )
            runs.append((init_idx, bb_params, val_loss, tp_params, train_pvoros, history))
        per_lr_runs[lr] = runs
        best_val_loss = min(r[2] for r in runs)
        print(f"[BCE grid] lr={lr:.0e} | best val_bce_loss={best_val_loss:.4f}")

    best_lr = min(lr_grid, key=lambda lr: min(r[2] for r in per_lr_runs[lr]))
    runs_at_best_lr = per_lr_runs[best_lr]

    _, bb_params, bb_val_loss, _, _, bb_history = min(runs_at_best_lr, key=lambda r: r[2])
    _, _, _, tp_params, tp_train_pvoros, tp_history = max(runs_at_best_lr, key=lambda r: r[4])
    init_bce_baseline_params = {r[0]: r[1] for r in runs_at_best_lr}

    print(f"[BCE] Selected lr={best_lr:.0e} for shared BCE run | "
          f"bce_baseline val_bce_loss={bb_val_loss:.4f} | "
          f"bce_track_train_pvoros train_pvoros={tp_train_pvoros:.4f}")

    return {
        "bce_baseline": (bb_params, bb_val_loss, bb_history, best_lr),
        "bce_track_train_pvoros": (tp_params, tp_train_pvoros, tp_history, best_lr),
        "init_bce_baseline_params_by_idx": init_bce_baseline_params,
    }


def tune_and_train_pvoros(X_train, y_train, X_val, y_val, lr_grid=LR_GRID, n_inits=N_INITS,
                           seed=SPLIT_SEED, init_from_by_idx=None, model_label="pvoros"):
    """Grid-search LR for a pVOROS-loss run, independent of the BCE run's LR.
    The winning LR is the one whose best-of-N-inits val pVOROS is highest."""
    per_lr_runs = {}
    for lr in lr_grid:
        runs = []
        for init_idx in range(n_inits):
            init_seed = seed + init_idx
            init_from = None if init_from_by_idx is None else init_from_by_idx[init_idx]
            params, val_pv, history = train_pvoros_loss(
                X_train, y_train, X_val, y_val, lr=lr, seed=init_seed, init_from=init_from, verbose=False
            )
            runs.append((init_idx, params, val_pv, history))
        per_lr_runs[lr] = runs
        best_val_pv = max(r[2] for r in runs)
        print(f"[{model_label} grid] lr={lr:.0e} | best val_pvoros={best_val_pv:.4f}")

    best_lr = max(lr_grid, key=lambda lr: max(r[2] for r in per_lr_runs[lr]))
    _, params, val_pv, history = max(per_lr_runs[best_lr], key=lambda r: r[2])
    print(f"[{model_label}] Selected lr={best_lr:.0e} | val_pvoros={val_pv:.4f}")
    return params, val_pv, history, best_lr


# ---------------------------------------------------------------------------
# 4. Plotting
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--kappa_frac", type=float, default=KAPPA_FRAC)
    parser.add_argument("--min_fp", type=float, default=MIN_FP)
    parser.add_argument("--max_fp", type=float, default=MAX_FP)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    global ALPHA, KAPPA_FRAC, MIN_FP, MAX_FP
    args = parse_args()
    ALPHA, KAPPA_FRAC, MIN_FP, MAX_FP = args.alpha, args.kappa_frac, args.min_fp, args.max_fp
    print(f"pVOROS config: alpha={ALPHA} | kappa_frac={KAPPA_FRAC} | min_fp={MIN_FP} | max_fp={MAX_FP}")

    config_tag = f"a{ALPHA:g}_k{KAPPA_FRAC:g}_minfp{MIN_FP:.4g}_maxfp{MAX_FP:.4g}"
    results_dir = RESULTS_DIR / config_tag
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # Each model is selected across a grid of LR_GRID x N_INITS random param
    # inits by its own checkpoint metric: lower val BCE loss for bce_baseline,
    # higher train pVOROS for bce_track_train_pvoros, higher val pVOROS for the
    # two pVOROS-trained models. bce_baseline and bce_track_train_pvoros share
    # one physical training run, so they share one tuned LR (chosen by
    # bce_baseline's metric); pvoros and pvoros_from_bce_init each get their
    # own independently tuned LR. The data split/scaler stay fixed at SPLIT_SEED.
    print("\n" + "#" * 70)
    print("  Tuning shared BCE run (Models 1 & 5)")
    print("#" * 70)
    bce_result = tune_and_train_bce(X_train, y_train, X_val, y_val)
    bce_baseline_params, bce_val_loss, bce_baseline_history, bce_lr = bce_result["bce_baseline"]
    bce_track_train_pvoros_params, bce_train_pvoros, bce_track_history, _ = bce_result["bce_track_train_pvoros"]

    print("\n" + "#" * 70)
    print("  Tuning pVOROS run (Model 3, random init)")
    print("#" * 70)
    pvoros_params, pvoros_val_pv, pvoros_history, pvoros_lr = tune_and_train_pvoros(
        X_train, y_train, X_val, y_val, model_label="pvoros"
    )

    print("\n" + "#" * 70)
    print("  Tuning pVOROS run (Model 4, init from Model 1)")
    print("#" * 70)
    pvoros_from_bce_init_params, pvoros_from_bce_val_pv, pvoros_from_bce_init_history, pvoros_from_bce_lr = (
        tune_and_train_pvoros(
            X_train, y_train, X_val, y_val,
            init_from_by_idx=bce_result["init_bce_baseline_params_by_idx"],
            model_label="pvoros_from_bce_init",
        )
    )

    best_scores = {
        "bce_baseline": bce_val_loss,
        "bce_track_train_pvoros": bce_train_pvoros,
        "pvoros": pvoros_val_pv,
        "pvoros_from_bce_init": pvoros_from_bce_val_pv,
    }
    best_lrs = {
        "bce_baseline": bce_lr,
        "bce_track_train_pvoros": bce_lr,
        "pvoros": pvoros_lr,
        "pvoros_from_bce_init": pvoros_from_bce_lr,
    }
    model_params = {
        "bce_baseline": bce_baseline_params,
        "bce_track_train_pvoros": bce_track_train_pvoros_params,
        "pvoros": pvoros_params,
        "pvoros_from_bce_init": pvoros_from_bce_init_params,
    }
    model_history = {
        "bce_baseline": bce_baseline_history,
        "bce_track_train_pvoros": bce_track_history,
        "pvoros": pvoros_history,
        "pvoros_from_bce_init": pvoros_from_bce_init_history,
    }

    print("\n" + "=" * 70)
    print(f"BEST-OF-{N_INITS}-INITS VAL SCORES PER MODEL (after LR tuning over {LR_GRID})")
    print("=" * 70)
    for model_name, score in best_scores.items():
        print(f"{model_name:24s} best_lr={best_lrs[model_name]:.0e} | best_val_score={score:.4f}")

    plot_model_traces(model_history["bce_baseline"], "bce_baseline", results_dir, loss_label="BCE Loss")
    plot_model_traces(model_history["bce_track_train_pvoros"], "bce_track_train_pvoros", results_dir, loss_label="BCE Loss")
    plot_model_traces(model_history["pvoros"], "pvoros", results_dir, loss_label="Soft PV Loss")
    plot_model_traces(model_history["pvoros_from_bce_init"], "pvoros_from_bce_init", results_dir, loss_label="Soft PV Loss")

    # Evaluate all models on test set.
    rows = []
    for model_name, params in model_params.items():
        test_pvoros = compute_pvoros_metric(params, x_test_jax, y_test) * 100
        y_test_pred = np.asarray(jax.nn.sigmoid(jnp.dot(x_test_jax, params["w"]) + params["b"]))
        test_auroc = plot_model_roc(y_test, y_test_pred, model_name, results_dir)

        rows.append({
            "model_name": model_name,
            "best_lr": best_lrs[model_name],
            "best_val_score": best_scores[model_name],
            "test_pvoros_pct": test_pvoros,
            "test_auroc": test_auroc,
        })

        np.save(results_dir / f"{model_name}_w.npy", np.asarray(params["w"]))
        np.save(results_dir / f"{model_name}_b.npy", np.asarray(params["b"]))

    results_df = pd.DataFrame(rows)
    results_csv_path = results_dir / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST SET EVALUATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved results table: {results_csv_path}")


if __name__ == "__main__":
    main()
