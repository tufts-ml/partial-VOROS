import numpy as np
import os
import jax
import jax.numpy as jnp
import optax
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from metrics_jax import pvoros_score, pv_loss_fixed_thresh
from train_busi_pca import (
    split_train_val_test, 
    load_embeddings_and_labels, 
    init_params, 
    compute_pvoros_metric, 
    bce_loss_fn
)
from pathlib import Path
import pandas as pd

DATA_DIR = Path("busi_training/busi_embeddings")
RESULTS_DIR = Path("busi_training/results")
GRIDSEARCH_DIR = RESULTS_DIR / "gs_a0.4_k1.0_min1-9_max1-6"

VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 0


def load_all_validation_results():
    """Load gridsearch results (pVOROS scores on validation)"""
    csv_files = list(GRIDSEARCH_DIR.glob("cv_val_results_w*_s*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No validation CSV files found in {GRIDSEARCH_DIR}")
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def get_best_hyperparameters(master_df):
    """Identifies optimal hyperparameter pairs per representation and method."""
    methods = {
        "PV (Random Init)": "val_pv_rand_score",
        "PV (BCE Init)": "val_pv_bce_score",
        "BCE (Std Val BCE)": "val_bce_score",
        "BCE (Monitored PV)": "val_bce_checkpoint_score",
    }
    best_configs = {}
    for ds in master_df["dataset"].unique():
        ds_df = master_df[master_df["dataset"] == ds]
        best_configs[ds] = {}
        for method_name, col in methods.items():
            best_idx = ds_df[col].idxmax()
            best_row = ds_df.loc[best_idx]
            best_configs[ds][method_name] = {
                "w": best_row["weight_decay"],
                "s": best_row["learning_rate"],
                "val_score": best_row[col],
            }
    return best_configs


def fit_and_train_final_model_with_checkpointing(
    X_tr, y_tr, X_va, y_va, method_type, lr, weight_decay, alpha, kappa_frac, min_fp, max_fp, epochs=100
):
    """Retrains on Train (60%) and monitors Val (20%) to save the best checkpoint."""
    x_tr, y_tr_j = jnp.asarray(X_tr, dtype=jnp.float64), jnp.asarray(y_tr, dtype=jnp.float64)
    x_va, y_va_j = jnp.asarray(X_va, dtype=jnp.float64), jnp.asarray(y_va, dtype=jnp.float64)
    P_tr, N_tr = jnp.sum(y_tr_j == 1.0), jnp.sum(y_tr_j == 0.0)
    kappa_tr = kappa_frac * (P_tr + N_tr)

    # ------------------------------------------------------------------
    # Method 3 & 4: Standard BCE and Monitored BCE
    # ------------------------------------------------------------------
    if method_type in ["bce_std", "bce_monitored"]:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))
        params = init_params(jax.random.PRNGKey(SPLIT_SEED), x_tr.shape[1])
        opt_state = optimizer.init(params)
        
        best_params = params
        best_metric = float("inf") if method_type == "bce_std" else -float("inf")

        for ep in range(1, epochs + 1):
            _, grads = jax.value_and_grad(bce_loss_fn)(params, x_tr, y_tr_j)
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)

            if method_type == "bce_std":
                va_bce_loss = float(bce_loss_fn(params, x_va, y_va_j))
                if va_bce_loss < best_metric:
                    best_metric = va_bce_loss
                    best_params = params
            elif method_type == "bce_monitored" and ep % 10 == 0:
                va_pv = compute_pvoros_metric(params, x_va, y_va, alpha, kappa_frac, min_fp, max_fp)
                if va_pv > best_metric:
                    best_metric = va_pv
                    best_params = params

        return best_params

    # ------------------------------------------------------------------
    # Method 1 & 2: Soft PV Loss (Random vs BCE Init)
    # ------------------------------------------------------------------
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr, weight_decay=weight_decay))

    if method_type == "pv_bce":
        bce_init = fit_and_train_final_model_with_checkpointing(
            X_tr, y_tr, X_va, y_va, "bce_std", lr, weight_decay, alpha, kappa_frac, min_fp, max_fp, epochs
        )
        params = {"w": jnp.asarray(bce_init["w"]), "b": jnp.asarray(bce_init["b"])}
    else:
        params = init_params(jax.random.PRNGKey(SPLIT_SEED), x_tr.shape[1])

    opt_state = optimizer.init(params)
    best_params = params
    best_val_pv = compute_pvoros_metric(params, x_va, y_va, alpha, kappa_frac, min_fp, max_fp)

    def pure_loss_fn(p):
        return pv_loss_fixed_thresh(p, x_tr, y_tr_j, P_tr, N_tr, kappa_tr, alpha, min_fp, max_fp)

    @jax.jit
    def train_step(p, state):
        loss, grads = jax.value_and_grad(pure_loss_fn)(p)
        updates, state = optimizer.update(grads, state, params=p)
        return optax.apply_updates(p, updates), state

    for ep in range(1, epochs + 1):
        params, opt_state = train_step(params, opt_state)
        if ep % 10 == 0:
            va_pv = compute_pvoros_metric(params, x_va, y_va, alpha, kappa_frac, min_fp, max_fp)
            if va_pv > best_val_pv:
                best_val_pv = va_pv
                best_params = params

    return best_params


def eval_test():
    print("=" * 80)
    print("      AGGREGATING CV HYPERPARAMETERS & RETRAINING WITH VAL CHECKPOINTING")
    print("=" * 80)

    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)
    
    # Standard 60% Train, 20% Val, 20% Test Split
    X_train_val, X_test_raw, y_train_val, y_test = split_train_val_test(all_feats, all_labels)
    relative_val_frac = VAL_FRACTION / (1.0 - TEST_FRACTION)
    
    X_train_raw, X_val_raw, y_train, y_val = split_train_val_test(
        X_train_val, y_train_val, val_frac=relative_val_frac, test_frac=0.0, seed=SPLIT_SEED
    )[:4]

    master_df = load_all_validation_results()
    best_configs = get_best_hyperparameters(master_df)

    alpha, kappa_frac, min_fp, max_fp = 0.6, 0.5, 1 / 9, 1 / 6
    pca_dimensions = [None, 2, 30, 120]
    test_summary = []

    for dim in pca_dimensions:
        ds_name = f"Full ({X_train_raw.shape[1]}D)" if dim is None else f"PCA {dim}D"
        cfg = best_configs[ds_name]

        # Fit transformers strictly on 60% Train
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(X_train_raw)
        X_va_proc = scaler.transform(X_val_raw)
        X_te_proc = scaler.transform(X_test_raw)

        if dim is not None:
            pca = PCA(n_components=dim, random_state=42)
            X_tr_proc = pca.fit_transform(X_tr_proc)
            X_va_proc = pca.transform(X_va_proc)
            X_te_proc = pca.transform(X_te_proc)

        x_test_jax = jnp.asarray(X_te_proc, dtype=jnp.float64)

        # 1. Method 1: PV Random Init
        pv_rand_params = fit_and_train_final_model_with_checkpointing(
            X_tr_proc, y_train, X_va_proc, y_val, "pv_rand", 
            cfg['PV (Random Init)']['s'], cfg['PV (Random Init)']['w'],
            alpha, kappa_frac, min_fp, max_fp
        )
        score_pv_rand = compute_pvoros_metric(pv_rand_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)

        # 2. Method 2: PV BCE Init
        pv_bce_params = fit_and_train_final_model_with_checkpointing(
            X_tr_proc, y_train, X_va_proc, y_val, "pv_bce", 
            cfg['PV (BCE Init)']['s'], cfg['PV (BCE Init)']['w'],
            alpha, kappa_frac, min_fp, max_fp
        )
        score_pv_bce = compute_pvoros_metric(pv_bce_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)

        # 3. Method 3: BCE Standard
        bce_std_params = fit_and_train_final_model_with_checkpointing(
            X_tr_proc, y_train, X_va_proc, y_val, "bce_std", 
            cfg['BCE (Std Val BCE)']['s'], cfg['BCE (Std Val BCE)']['w'],
            alpha, kappa_frac, min_fp, max_fp
        )
        score_bce_std = compute_pvoros_metric(bce_std_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)

        # 4. Method 4: BCE Monitored
        bce_mon_params = fit_and_train_final_model_with_checkpointing(
            X_tr_proc, y_train, X_va_proc, y_val, "bce_monitored", 
            cfg['BCE (Monitored PV)']['s'], cfg['BCE (Monitored PV)']['w'],
            alpha, kappa_frac, min_fp, max_fp
        )
        score_bce_mon = compute_pvoros_metric(bce_mon_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)

        test_summary.append({
            "Representation": ds_name,
            "PV (Rand) Test %": f"{score_pv_rand * 100:.2f}%",
            "PV (BCE Init) Test %": f"{score_pv_bce * 100:.2f}%",
            "BCE (Std) Test %": f"{score_bce_std * 100:.2f}%",
            "BCE (Monitored) Test %": f"{score_bce_mon * 100:.2f}%",
        })

    df_test = pd.DataFrame(test_summary)
    test_out_path = RESULTS_DIR / "gridsearch_test_evaluation_summary.csv"
    df_test.to_csv(test_out_path, index=False)

    print("\n" + "=" * 105)
    print("                               FINAL HELD-OUT TEST SET EVALUATION TABLE")
    print("=" * 105)
    print(f"{'Representation':<22} | {'PV (Random Init)':<18} | {'PV (BCE Init)':<18} | {'BCE (Std Val BCE)':<18} | {'BCE (Monitored PV)':<18}")
    print("-" * 105)
    for row in test_summary:
        print(
            f"{row['Representation']:<22} | "
            f"{row['PV (Rand) Test %']:<18} | "
            f"{row['PV (BCE Init) Test %']:<18} | "
            f"{row['BCE (Std) Test %']:<18} | "
            f"{row['BCE (Monitored) Test %']:<18}"
        )
    print("=" * 105)


if __name__ == "__main__":
    eval_test()