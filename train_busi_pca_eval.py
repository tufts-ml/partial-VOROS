import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from metrics_jax import pvoros_score, pv_loss_fixed_thresh, soft_roc_fixed_thresholds
from metrics import pvoros_score as pvoros_score_np
from train_busi_pca import split_train_val_test, load_embeddings_and_labels, compute_pvoros_metric
import _geometry_jax
from pathlib import Path
import argparse
import pandas as pd

DATA_DIR = Path("busi_training/busi_embeddings")
RESULTS_DIR = Path("busi_training/results")
RESULTS_DIR.mkdir(exist_ok=True)

GRIDSEARCH_DIR = RESULTS_DIR / "gridsearch"
GRIDSEARCH_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 0


def load_all_validation_results():
    csv_files = list(GRIDSEARCH_DIR.glob("val_results_w*_i*_s*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No validation CSV files found in {GRIDSEARCH_DIR}")

    dfs = [pd.read_csv(f) for f in csv_files]
    master_df = pd.concat(dfs, ignore_index=True)
    return master_df


def get_best_hyperparameters(master_df):
    """Identifies top hyperparameters per representation and method."""
    methods = {
        "PV (Random Init)": "val_pv_rand_score",
        "PV (BCE Init)": "val_pv_bce_score",
        "BCE (Std Val BCE)": "val_bce_score",
        "BCE (Monitored PV)": "val_bce_checkpoint_score",
    }

    best_configs = {}
    datasets = master_df["dataset"].unique()

    for ds in datasets:
        ds_df = master_df[master_df["dataset"] == ds]
        best_configs[ds] = {}

        for method_name, col in methods.items():
            best_idx = ds_df[col].idxmax()
            best_row = ds_df.loc[best_idx]
            best_configs[ds][method_name] = {
                "w": best_row["weight_decay"],
                "i": int(best_row["epochs"]),
                "s": best_row["learning_rate"],
                "val_score": best_row[col],
            }

    return best_configs

def eval_test():
    print("=" * 80)
    print("          READING VALIDATION RESULTS & EVALUATING HELD-OUT TEST SET")
    print("=" * 80)

    # Load data
    all_feats, all_labels = load_embeddings_and_labels(DATA_DIR)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_val_test(all_feats, all_labels)

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

    # 1. Gather best hyperparameter configs from all val_*.csv files
    master_df = load_all_validation_results()
    best_configs = get_best_hyperparameters(master_df)

    alpha = 0.6
    kappa_frac = 0.5
    min_fp = 1 / 9
    max_fp = 1 / 6

    test_summary = []

    # 2. Evaluate models on held-out test set
    for name, (X_train, X_val, X_test) in datasets.items():
        dim_label = name.split()[1] if "PCA" in name else "full"
        x_test_jax = jnp.asarray(X_test, dtype=jnp.float64)
        cfg = best_configs[name]

        # Load weights saved during best checkpoints
        best_w = cfg['PV (Random Init)']['w']
        best_i = cfg['PV (Random Init)']['i']
        best_s = cfg['PV (Random Init)']['s']

        param_tag = f"{dim_label}_w{best_w}_i{best_i}_s{best_s}"
        pv_rand_w = np.load(RESULTS_DIR / f"pv_rand_w_{param_tag}.npy")
        pv_rand_b = np.load(RESULTS_DIR / f"pv_rand_b_{param_tag}.npy")
        pv_rand_params = {"w": jnp.asarray(pv_rand_w), "b": jnp.asarray(pv_rand_b)}


        best_w = cfg['PV (BCE Init)']['w']
        best_i = cfg['PV (BCE Init)']['i']
        best_s = cfg['PV (BCE Init)']['s']

        param_tag = f"{dim_label}_w{best_w}_i{best_i}_s{best_s}"
        pv_bce_w = np.load(RESULTS_DIR / f"pv_bce_w_{param_tag}.npy")
        pv_bce_b = np.load(RESULTS_DIR / f"pv_bce_b_{param_tag}.npy")
        pv_bce_params = {"w": jnp.asarray(pv_bce_w), "b": jnp.asarray(pv_bce_b)}


        best_w = cfg['BCE (Std Val BCE)']['w']
        best_i = cfg['BCE (Std Val BCE)']['i']
        best_s = cfg['BCE (Std Val BCE)']['s']

        param_tag = f"{dim_label}_w{best_w}_i{best_i}_s{best_s}"
        bce_std_w = np.load(RESULTS_DIR / f"bce_std_w_{param_tag}.npy")
        bce_std_b = np.load(RESULTS_DIR / f"bce_std_b_{param_tag}.npy")
        bce_std_params = {"w": jnp.asarray(bce_std_w), "b": jnp.asarray(bce_std_b)}


        best_w = cfg['BCE (Monitored PV)']['w']
        best_i = cfg['BCE (Monitored PV)']['i']
        best_s = cfg['BCE (Monitored PV)']['s']

        param_tag = f"{dim_label}_w{best_w}_i{best_i}_s{best_s}"
        bce_mon_w = np.load(RESULTS_DIR / f"bce_monitored_w_{param_tag}.npy")
        bce_mon_b = np.load(RESULTS_DIR / f"bce_monitored_b_{param_tag}.npy")
        bce_mon_params = {"w": jnp.asarray(bce_mon_w), "b": jnp.asarray(bce_mon_b)}

        # Evaluate pVOROS score on Test Set
        score_pv_rand = compute_pvoros_metric(pv_rand_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)
        score_pv_bce = compute_pvoros_metric(pv_bce_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)
        score_bce_std = compute_pvoros_metric(bce_std_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)
        score_bce_mon = compute_pvoros_metric(bce_mon_params, x_test_jax, y_test, alpha, kappa_frac, min_fp, max_fp)

        
        test_summary.append({
            "Representation": name,
            "PV (Rand) Test %": f"{score_pv_rand * 100:.2f}%",
            "PV (Rand) Best Params": f"w={cfg['PV (Random Init)']['w']}, s={cfg['PV (Random Init)']['s']}, i={cfg['PV (Random Init)']['i']}",
            "PV (BCE Init) Test %": f"{score_pv_bce * 100:.2f}%",
            "PV (BCE Init) Best Params": f"w={cfg['PV (BCE Init)']['w']}, s={cfg['PV (BCE Init)']['s']}, i={cfg['PV (BCE Init)']['i']}",
            "BCE (Std) Test %": f"{score_bce_std * 100:.2f}%",
            "BCE (Std) Best Params": f"w={cfg['BCE (Std Val BCE)']['w']}, s={cfg['BCE (Std Val BCE)']['s']}, i={cfg['BCE (Std Val BCE)']['i']}",
            "BCE (Monitored) Test %": f"{score_bce_mon * 100:.2f}%",
            "BCE (Monitored) Best Params": f"w={cfg['BCE (Monitored PV)']['w']}, s={cfg['BCE (Monitored PV)']['s']}, i={cfg['BCE (Monitored PV)']['i']}",
        })

    # Save Test Summary Table
    df_test = pd.DataFrame(test_summary)
    test_out_path = RESULTS_DIR / "gridsearch_test_evaluation_summary.csv"
    df_test.to_csv(test_out_path, index=False)

    # Print Final Summary Table to stdout
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
    print(f"\nFull test report with hyperparameter breakdown saved to: {test_out_path}\n")


if __name__ == "__main__":
    eval_test()