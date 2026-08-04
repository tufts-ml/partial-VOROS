import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from metrics_jax import pvoros_score, pv_loss_fixed_thresh
from pathlib import Path
from train_busi import get_or_build_embeddings

DATA_DIR = Path("busi_data")       # expects DATA_DIR/{benign,malignant,normal}/*.png
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 32
VAL_FRACTION = 0.2
SPLIT_SEED = 0

LR = 1e-4       # Lower learning rate for Partial VOROS
EPOCHS = 50

# Assuming EPOCHS, LR, VAL_FRACTION, SPLIT_SEED, CACHE_DIR, DATA_DIR, pv_loss are defined above

def split_train_val(feats, labels, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    return train_test_split(
        feats, labels,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# 3. Dynamic & Static Score Thresholding
# ---------------------------------------------------------------------------
def init_params(key, dim):
    return {
        "w": jax.random.normal(key, (dim,), dtype=jnp.float64) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float64),
    }


def train_logreg(feats, labels, epochs=EPOCHS, lr=LR, seed=0, n_restarts=5, inits_per_seed=10):
    x = jnp.asarray(feats, dtype=jnp.float64)
    y = jnp.asarray(labels, dtype=jnp.float64)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )

    P = jnp.sum(y == 1.0)
    N = jnp.sum(y == 0.0)
    kappa = 0.5 * (P + N)
    alpha = 0.3
    min_fp_cost_ratio = 1 / 9
    max_fp_cost_ratio = 1 / 6
    n_points = 1000
    temp = 0.02

    def loss_fn(p):
        return pv_loss_fixed_thresh(
            p, x, y, P, N, kappa, alpha,
            min_fp_cost_ratio, max_fp_cost_ratio, n_points, temp
        )

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    def run_single_init(init_key):
        params = init_params(init_key, x.shape[1])
        opt_state = optimizer.init(params)

        best_params = params
        best_loss = float("inf")

        for epoch in range(epochs):
            params, opt_state, loss, grads = train_step(params, opt_state)
            curr_loss = float(loss)

            if curr_loss < best_loss and not np.isnan(curr_loss):
                best_loss = curr_loss
                best_params = params

        return best_params, best_loss

    best_overall_params = None
    best_overall_loss = float("inf")

    # Outer loop: External Seeds
    for i in range(n_restarts):
        run_seed = seed + i
        seed_key = jax.random.PRNGKey(run_seed)
        
        # Derive 10 distinct subkeys for weight initializations under this seed
        init_keys = jax.random.split(seed_key, inits_per_seed)

        print(f"\n--- Seed {run_seed} ({i + 1}/{n_restarts}) | Training {inits_per_seed} Weight Inits ---")
        
        seed_best_params = None
        seed_best_loss = float("inf")

        # Inner loop: 10 random weight initializations per seed
        for init_idx, k in enumerate(init_keys):
            params, final_loss = run_single_init(k)
            print(f"  └─ Init {init_idx + 1:2d}/{inits_per_seed} -> Loss: {final_loss:.4f}")

            if final_loss < seed_best_loss:
                seed_best_loss = final_loss
                seed_best_params = params

        print(f">> Best Loss for Seed {run_seed}: {seed_best_loss:.4f}")

        if seed_best_loss < best_overall_loss:
            best_overall_loss = seed_best_loss
            best_overall_params = seed_best_params

    print(f"\n[SUMMARY] Best Overall Loss across all seeds & inits: {best_overall_loss:.4f}")
    return best_overall_params


# ---------------------------------------------------------------------------
# 4. Main Pipeline (Loops across Full & PCA Reduced Dimensions)
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = get_or_build_embeddings(DATA_DIR)

    # Split train/val ONCE before applying PCA to avoid data leakage
    X_train_raw, X_val_raw, y_train, y_val = split_train_val(all_feats, all_labels)

    print(f"Train samples: {X_train_raw.shape[0]}, Malignant rate: {y_train.mean():.3f}")
    print(f"Val samples:   {X_val_raw.shape[0]}, Malignant rate: {y_val.mean():.3f}")

    # Define target PCA dimensions to iterate over (alongside original full dimensions)
    pca_dimensions = [2, 10, 20]
    
    # Pre-compute reduced representations for Train & Val
    datasets = {
        f"Full ({X_train_raw.shape[1]}D)": (X_train_raw, X_val_raw)
    }

    for dim in pca_dimensions:
        pca = PCA(n_components=dim, random_state=42)
        # Fit PCA exclusively on train set, then transform both train and val
        X_train_pca = pca.fit_transform(X_train_raw)
        X_val_pca = pca.transform(X_val_raw)
        
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        datasets[f"PCA {dim}D ({explained_var:.1f}% var)"] = (X_train_pca, X_val_pca)

    results_summary = {}

    # Iterate through each feature dimensionality configuration
    for name, (X_train, X_val) in datasets.items():
        print("\n" + "=" * 65)
        print(f"        RUNNING EXPERIMENT: {name}")
        print("=" * 65)

        params = train_logreg(X_train, y_train)

        x_val = jnp.asarray(X_val, dtype=jnp.float64)
        y_val_jax = jnp.asarray(y_val, dtype=jnp.float64)

        P_val = jnp.sum(y_val_jax == 1.0)
        N_val = jnp.sum(y_val_jax == 0.0)
        kappa_val = 0.5 * (P_val + N_val)
        alpha = 0.3
        min_fp_cost_ratio = 1 / 9
        max_fp_cost_ratio = 1 / 6
        n_points = 1000

        logits_val = jnp.dot(x_val, params["w"]) + params["b"]
        y_pred_val = jax.nn.sigmoid(logits_val)

        # Diagnostics
        print("\n--- VALIDATION PREDICTIONS DIAGNOSTIC ---")
        print(f"y_pred_val min : {float(jnp.min(y_pred_val)):.6f}")
        print(f"y_pred_val max : {float(jnp.max(y_pred_val)):.6f}")
        print(f"y_pred_val mean: {float(jnp.mean(y_pred_val)):.6f}")
        print(f"y_pred_val std : {float(jnp.std(y_pred_val)):.6f}")

        # Static Grid Score
        pv_fixed_thresh = pv_loss_fixed_thresh(
            params, 
            x_val, 
            y_val_jax, 
            P_val, 
            N_val, 
            kappa_val, 
            alpha,
            min_fp_cost_ratio, 
            max_fp_cost_ratio, 
            n_points)
        pv_fixed_thresh = -float(pv_fixed_thresh)

        # Empirical ROC VOROS Score
        fprs_emp, tprs_emp, _ = roc_curve(np.asarray(y_val), np.asarray(y_pred_val))
        pv_score = pvoros_score(
            y_true=y_val,
            y_pred=y_pred_val,
            alpha=alpha,
            kappa_frac=0.5,
            min_fp_cost_ratio=min_fp_cost_ratio,
            max_fp_cost_ratio=max_fp_cost_ratio,
            n_points=n_points
        )

        # Plot empirical ROC with alpha and kappa constraint bounds
        P = int(np.sum(y_val == 1))
        N = int(np.sum(y_val == 0))
        prevalence = P / (P + N)
        alpha_slope = alpha * (1 - prevalence) / (prevalence * (1 - alpha))
        kappa_slope = -(N / P)
        kappa_plot = 0.5 * (P + N)

        fpr_grid = np.linspace(0.0, 1.0, 200)
        tpr_alpha_bound = alpha_slope * fpr_grid
        tpr_kappa_bound = kappa_slope * fpr_grid + (kappa_plot / P)

        plt.figure(figsize=(8, 8))
        plt.plot(fprs_emp, tprs_emp, color='#1f77b4', lw=2.5, label='Empirical ROC')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance Baseline')
        plt.plot(fpr_grid, tpr_alpha_bound, color='#d62728', linestyle='--', lw=2.0, label=f'Alpha Bound (slope={alpha_slope:.2f})')
        plt.plot(fpr_grid, tpr_kappa_bound, color='#2ca02c', linestyle='--', lw=2.0, label=f'Kappa Bound (slope={kappa_slope:.2f})')

        y_lower_clamped = np.clip(tpr_alpha_bound, 0.0, 1.0)
        y_upper_clamped = np.clip(tpr_kappa_bound, 0.0, 1.0)
        plt.fill_between(
            fpr_grid,
            y_lower_clamped,
            y_upper_clamped,
            where=(y_upper_clamped >= y_lower_clamped),
            color='#ff7f0e', alpha=0.18, label='Feasible Region'
        )

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC with Alpha/Kappa Bounds: {name}', fontsize=13)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        plot_name = CACHE_DIR / f'roc_bounds_{name.replace(" ", "_").replace("/", "_")}.pdf'
        plt.savefig(plot_name, format='pdf', dpi=300)
        plt.close()

        results_summary[name] = {
            "pv_fixed_thresh": pv_fixed_thresh * 100,
            "pv_score": float(pv_score) * 100,
        }

        # Cache final model weights per dimension
        dim_label = name.split()[1] if "PCA" in name else "full"
        np.save(CACHE_DIR / f"logreg_w_{dim_label}.npy", np.asarray(params["w"]))
        np.save(CACHE_DIR / f"logreg_b_{dim_label}.npy", np.asarray(params["b"]))

    # Final summary banner printout
    print("\n" + "=" * 65)
    print("            FINAL DIMENSION COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Representation':<25} | {'PV fixed thresh (%)':<18} | {'PV score (%)':<22}")
    print("-" * 65)
    for name, metrics in results_summary.items():
        print(f"{name:<25} | {metrics['pv_fixed_thresh']:18.2f} | {metrics['pv_score']:22.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()