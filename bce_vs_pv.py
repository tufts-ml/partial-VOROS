"""
For each seed dataset:
  1. Load data_dict['data']['x'] / ['y'] and split into train/val (80/20).
  2. Train a logistic regression model with plain BCE loss (10 restarts, pick best).
  3. Train a second logistic regression model with the PVOROS loss (10 restarts, pick best).
  4. Evaluate both best-trained models' PVOROS score on the held-out validation split.
  5. Print a summary table comparing the two.
"""

import os
# Force deterministic CPU execution before JAX loads
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import optax

from metrics_jax import pv_loss_theta_c, pvoros_score

# ---- Shared constants (reused from the earlier test file) ----
KAPPA_FRAC = 0.5
ALPHA = 0.6
MIN_FP_COST_RATIO = 1 / 9
MAX_FP_COST_RATIO = 1 / 6
N_POINTS = 1000

LEARNING_RATE = 0.01
N_EPOCHS = 100
VAL_FRACTION = 0.2
N_RESTARTS = 10  # Number of random initializations

SEED_FILENAMES = [
    "seed_101_201.npy",
    "seed_301_101.npy",
    "seed_501_801.npy",
    "seed_601_201.npy",
    "seed_701_501.npy",
]


def load_seed_data(seed_filename):
    """Load data from a seed file."""
    data_dict = np.load(seed_filename, allow_pickle=True).item()
    x = data_dict["data"]["x"]
    y = data_dict["data"]["y"]
    return x, y


def train_val_split(x, y, seed_filename, val_fraction=VAL_FRACTION):
    """100% deterministic 80/20 split across Python sessions using MD5 hashing."""
    seed_int = int(hashlib.md5(seed_filename.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed=seed_int)
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (
        np.asarray(x)[train_idx], np.asarray(y)[train_idx],
        np.asarray(x)[val_idx], np.asarray(y)[val_idx],
    )


def init_params(d, key):
    w_key, _ = jax.random.split(key)
    return {
        "w": jax.random.normal(w_key, (d,), dtype=jnp.float32) * 0.01,
        "b": jnp.array(0.0, dtype=jnp.float32),
    }


def init_params_theta_c(key):
    theta_key, c_key = jax.random.split(key)
    return {
        "theta": jax.random.uniform(theta_key, (), minval=0.0, maxval=2 * jnp.pi, dtype=jnp.float32),
        "c": jax.random.uniform(c_key, (), minval=-1.0, maxval=1.0, dtype=jnp.float32),
    }
 
 
def theta_c_to_wb(theta, c, M=1.0):
    """Angular (theta, c) -> linear boundary (w, b). Only valid for 2D features."""
    w1 = M * jnp.sin(theta)
    w2 = -M * jnp.cos(theta)
    w_vec = jnp.array([w1, w2], dtype=jnp.float32)
    b_val = M * c * jnp.cos(theta)
    return w_vec, b_val


def bce_loss(params, X, y):
    logits = jnp.dot(X, params["w"]) + params["b"]
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()


def make_bce_step(optimizer):
    @jax.jit
    def step(params, opt_state, X, y):
        loss, grads = jax.value_and_grad(bce_loss)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


def make_pvoros_step(
    optimizer, 
    P, 
    N, 
    kappa, 
    alpha,
    min_fp_cost_ratio, 
    max_fp_cost_ratio, 
    n_points):
    def loss_fn(params, X, y):
        loss = pv_loss_theta_c(params, X, y, P, N, kappa, alpha, min_fp_cost_ratio, max_fp_cost_ratio, n_points)
        return loss

    @jax.jit
    def step(params, opt_state, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


def train_bce(
    X_train, 
    y_train, 
    d, 
    key, 
    n_epochs=N_EPOCHS, 
    lr=LEARNING_RATE,
    ):
    params = init_params(d, key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    step = make_bce_step(optimizer)

    loss = 0.0
    for epoch in range(n_epochs):
        params, opt_state, loss = step(params, opt_state, X_train, y_train)

    return params, float(loss)


def train_pvoros(
    X_train, 
    y_train, 
    d, 
    key, 
    P, 
    N, 
    kappa,
    n_epochs=N_EPOCHS, 
    lr=LEARNING_RATE,
    ):
    params = init_params_theta_c(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    step = make_pvoros_step(
        optimizer, 
        P, 
        N, 
        kappa, 
        ALPHA,
        MIN_FP_COST_RATIO, 
        MAX_FP_COST_RATIO, 
        N_POINTS,
    )

    loss = 0.0
    for epoch in range(n_epochs):
        params, opt_state, loss = step(params, opt_state, X_train, y_train)

    return params, float(loss)


def eval_pvoros_score(params, X_val, y_val, P, N, kappa):
    """Return the (positive) PVOROS score on validation data."""
    w = params['w']
    b = params['b']

    logits = jnp.dot(X_val, w) + b
    y_pred = jax.nn.sigmoid(logits)

    score = pvoros_score(y_val, y_pred, ALPHA, kappa, MIN_FP_COST_RATIO, MAX_FP_COST_RATIO)
    return float(score)


def eval_pvoros_score_theta_c(params, X_val, y_val, P, N, kappa):
    w_vec, b_val = theta_c_to_wb(params['theta'], params['c'])
    wb_params = {"w": w_vec, "b": b_val}
    return eval_pvoros_score(wb_params, X_val, y_val, P, N, kappa)


def main():
    # Initialize global numpy rng and jax PRNGKey
    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    results = []
 
    for seed_filename in SEED_FILENAMES:
        x, y = load_seed_data(seed_filename)
        X_train, y_train, X_val, y_val = train_val_split(x, y, seed_filename)
 
        X_train = jnp.asarray(X_train, dtype=jnp.float32)
        y_train = jnp.asarray(y_train, dtype=jnp.float32)
        X_val = jnp.asarray(X_val, dtype=jnp.float32)
        y_val = jnp.asarray(y_val, dtype=jnp.float32)
 
        d = X_train.shape[1]
 
        P_train = float(jnp.sum(y_train == 1.0))
        N_train = float(jnp.sum(y_train == 0.0))
        KAPPA_train = KAPPA_FRAC * (P_train + N_train)
 
        P_val = float(jnp.sum(y_val == 1.0))
        N_val = float(jnp.sum(y_val == 0.0))
        KAPPA_val = KAPPA_FRAC * (P_val + N_val)

        print(f"\n[{seed_filename}] Training best of {N_RESTARTS} initializations...")

        best_bce_params = None
        best_bce_loss = float('inf')

        best_pvoros_params = None
        best_pvoros_loss = float('inf')
 
        # Loop over N restarts for both models
        for i in range(N_RESTARTS):
            key, bce_key, pvoros_key = jax.random.split(key, 3)

            # 1. Train BCE 
            bce_params, bce_loss_val = train_bce(X_train, y_train, d, bce_key)
            if bce_loss_val < best_bce_loss:
                best_bce_loss = bce_loss_val
                best_bce_params = bce_params

            # 2. Train PVOROS (Remember: lower loss == higher VOROS score)
            pvoros_params, pvoros_loss_val = train_pvoros(
                X_train, y_train, d, pvoros_key, P_train, N_train, KAPPA_train
            )
            if pvoros_loss_val < best_pvoros_loss:
                best_pvoros_loss = pvoros_loss_val
                best_pvoros_params = pvoros_params
                
        print(f"[{seed_filename}] -> Best BCE Train Loss:    {best_bce_loss:.6f}")
        print(f"[{seed_filename}] -> Best PVOROS Train Loss: {best_pvoros_loss:.6f}")
 
        # Evaluate best-performing parameters on the held-out validation set
        bce_val_score = eval_pvoros_score(
            best_bce_params, X_val, y_val, P_val, N_val, KAPPA_val
        )
        pvoros_val_score = eval_pvoros_score_theta_c(
            best_pvoros_params, X_val, y_val, P_val, N_val, KAPPA_val
        )
 
        results.append({
            "seed": seed_filename,
            "bce_val_pvoros": bce_val_score,
            "pvoros_val_pvoros": pvoros_val_score,
        })
 
    print("\n")
    print_results_table(results)
    return results


def print_results_table(results):
    header = f"{'seed':<20} {'BCE val PVOROS':>16} {'PVOROS-loss val PVOROS':>24}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['seed']:<20} {r['bce_val_pvoros']:>16.4f}"
            f"{r['pvoros_val_pvoros']:>24.4f}"
        )


if __name__ == "__main__":
    main()