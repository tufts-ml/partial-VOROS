import numpy as np
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

import jax
print("JAX Devices:", jax.devices())
print("JAX Default Backend:", jax.default_backend())

import jax.numpy as jnp
import optax



from metrics_jax import pv_loss
from pathlib import Path

from jax import config
config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("busi_data")       # expects DATA_DIR/{benign,malignant,normal}/*.png
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
VAL_FRACTION = 0.2
SPLIT_SEED = 0

LR = 1e-4       # Lower learning rate for Partial VOROS
EPOCHS = 200


# ---------------------------------------------------------------------------
# 1. Dataset & Preprocessing (Crop, Resize, 3-Class -> Binary Mapping)
# ---------------------------------------------------------------------------
LABEL_MAP = {"benign": 0, "normal": 0, "malignant": 1}

class UltrasoundDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths, self.labels = [], []
        for cls, label in LABEL_MAP.items():
            for p in (root / cls).glob("*"):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.paths.append(p)
                    self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L").convert("RGB")
        return self.transform(img), self.labels[idx]


def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ---------------------------------------------------------------------------
# 2. Frozen ViT Encoder & Embedding Cache
# ---------------------------------------------------------------------------
def build_encoder():
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).to("cuda")
    model.heads = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_embeddings(model, loader):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        emb = model(x)
        feats.append(emb.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


def get_or_build_embeddings(root: Path):
    feat_path = CACHE_DIR / "all_feats.npy"
    label_path = CACHE_DIR / "all_labels.npy"
    if feat_path.exists() and label_path.exists():
        print("DEBUG: Loading embeddings from cache.")
        feats = np.load(feat_path)
        labels = np.load(label_path)
        print(f"DEBUG: Cached all_feats.shape = {feats.shape}, all_labels.shape = {labels.shape}")
        return feats, labels

    transform = make_transform()
    ds = UltrasoundDataset(root, transform)
    if len(ds) == 0:
        raise RuntimeError(f"Found 0 images under {root}.")

    # Set num_workers=0 to prevent interactive thread freezes
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    encoder = build_encoder()
    feats, labels = extract_embeddings(encoder, loader)
    np.save(feat_path, feats)
    np.save(label_path, labels)
    print(f"DEBUG: Generated all_feats.shape = {feats.shape}, all_labels.shape = {labels.shape}")
    return feats, labels


def split_train_val(feats, labels, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    from sklearn.model_selection import train_test_split
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


def get_prediction_thresholds_dynamic(y_pred, num_thresholds=100):
    """Dynamic quantile thresholds for non-differentiable evaluation."""
    eps = 1e-5
    q = jnp.linspace(1.0 - eps, eps, num_thresholds)
    thresholds = jnp.quantile(y_pred, q)
    return jax.lax.stop_gradient(thresholds)


def train_logreg(feats, labels, epochs=EPOCHS, lr=LR, seed=0, n_restarts=5):
    x = jnp.asarray(feats, dtype=jnp.float64)
    y = jnp.asarray(labels, dtype=jnp.float64)

    # AdamW with weight decay and gradient norm clipping prevents logit explosion
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
    temp = 0.08  # Smoother temperature guarantees active non-zero gradients

    # Static 100-point grid for optimization prevents 'chasing thresholds' zero-gradient trap
    static_train_thresholds = jnp.linspace(0.001, 0.999, 100)

    def loss_fn(p):
        return pv_loss(
            p, x, y, P, N, kappa, alpha, static_train_thresholds,
            min_fp_cost_ratio, max_fp_cost_ratio, n_points, temp
        )

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    def run_single(run_seed):
        key = jax.random.PRNGKey(run_seed)
        params = init_params(key, x.shape[1])
        opt_state = optimizer.init(params)

        best_params = params
        best_loss = float("inf")

        for epoch in range(epochs):
            params, opt_state, loss, grads = train_step(params, opt_state)
            w_grad_norm = float(jnp.linalg.norm(grads["w"]))
            curr_loss = float(loss)

            if curr_loss < best_loss and not np.isnan(curr_loss):
                best_loss = curr_loss
                best_params = params

            if epoch % 25 == 0 or epoch == epochs - 1:
                print(f"[seed {run_seed:2d}] epoch {epoch:4d} | loss: {curr_loss:.4f} | w_grad_norm: {w_grad_norm:.8f}")

        return best_params, best_loss

    best_overall_params = None
    best_overall_loss = float("inf")

    for i in range(n_restarts):
        run_seed = seed + i
        print(f"\n--- Restart {i + 1}/{n_restarts} (Seed {run_seed}) ---")
        params, final_loss = run_single(run_seed)
        print(f"Result Seed {run_seed} -> Best Loss: {final_loss:.4f}")

        if final_loss < best_overall_loss:
            best_overall_loss = final_loss
            best_overall_params = params

    print(f"\nBest Overall Loss across restarts: {best_overall_loss:.4f}")
    return best_overall_params


# ---------------------------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = get_or_build_embeddings(DATA_DIR)

    X_train, X_val, y_train, y_val = split_train_val(all_feats, all_labels)

    print(f"Train samples: {X_train.shape[0]}, Malignant rate: {y_train.mean():.3f}")
    print(f"Val samples:   {X_val.shape[0]}, Malignant rate: {y_val.mean():.3f}")

    params = train_logreg(X_train, y_train)

    x_val = jnp.asarray(X_val, dtype=jnp.float64)
    y_val = jnp.asarray(y_val, dtype=jnp.float64)
    
    P_val = jnp.sum(y_val == 1.0)
    N_val = jnp.sum(y_val == 0.0)
    kappa_val = 0.5 * (P_val + N_val)
    alpha = 0.3
    min_fp_cost_ratio = 1 / 9
    max_fp_cost_ratio = 1 / 6
    n_points = 1000

    logits_val = jnp.dot(x_val, params["w"]) + params["b"]
    y_pred_val = jax.nn.sigmoid(logits_val)

    # Print prediction spread to check for logit saturation
    print("\n--- VALIDATION PREDICTIONS DIAGNOSTIC ---")
    print(f"y_pred_val min : {float(jnp.min(y_pred_val)):.6f}")
    print(f"y_pred_val max : {float(jnp.max(y_pred_val)):.6f}")
    print(f"y_pred_val mean: {float(jnp.mean(y_pred_val)):.6f}")
    print(f"y_pred_val std : {float(jnp.std(y_pred_val)):.6f}")

    from _geometry_jax import total_region_area, voros_jax
    from sklearn.metrics import roc_curve

    train_tot_area, _ = total_region_area(
        jnp.sum(y_train == 1.0), jnp.sum(y_train == 0.0), alpha, 0.5 * len(y_train)
    )
    val_tot_area, _ = total_region_area(
        P_val, N_val, alpha, kappa_val
    )

    print(f"\n[DIAGNOSTIC] Training Total Region Area  : {float(train_tot_area):.6f}")
    print(f"[DIAGNOSTIC] Validation Total Region Area: {float(val_tot_area):.6f}")

    # --- Method 1: Static Grid Thresholds ---
    static_val_thresholds = jnp.linspace(0.001, 0.999, 100)
    val_loss_static = pv_loss(
        params, x_val, y_val, P_val, N_val, kappa_val, alpha,
        static_val_thresholds, min_fp_cost_ratio, max_fp_cost_ratio, n_points, temp=0.08
    )
    score_static = -float(val_loss_static)

    # --- Method 2: Empirical Hard ROC Curve (bypasses sigmoid saturation) ---
    fprs_emp, tprs_emp, _ = roc_curve(np.asarray(y_val), np.asarray(y_pred_val))
    
    score_true = voros_jax(
        jnp.asarray(fprs_emp, dtype=jnp.float64),
        jnp.asarray(tprs_emp, dtype=jnp.float64),
        κ=kappa_val,
        α=alpha,
        P=P_val,
        N=N_val,
        min_fp_cost_ratio=min_fp_cost_ratio,
        max_fp_cost_ratio=max_fp_cost_ratio,
        n_points=n_points
    )

    print("\n" + "=" * 55)
    print("        VALIDATION PARTIAL VOROS COMPARISON")
    print("=" * 55)
    print(f"Static Soft Grid (temp=0.08) | Loss: {float(val_loss_static):.4f} | VOROS: {score_static * 100:.2f}%")
    print(f"True Hard Empirical ROC     | VOROS: {float(score_true) * 100:.2f}%")
    print("=" * 55)

    np.save(CACHE_DIR / "logreg_w.npy", np.asarray(params["w"]))
    np.save(CACHE_DIR / "logreg_b.npy", np.asarray(params["b"]))
    print("Saved trained head weights to cache.")



if __name__ == "__main__":
    main()