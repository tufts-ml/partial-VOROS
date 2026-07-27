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
import optaxgit rm --cached busi_data.zip

from metrics_jax import pv_loss
from pathlib import Path

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

LR = 1e-2
EPOCHS = 300


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
    # Standard ViT preprocessing with central crop and 224x224 resize
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
    model = vit_b_16(weights=weights).to("cuda") # Load directly to CUDA
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
        print(f"DEBUG: Cached all_labels min = {labels.min()}, max = {labels.max()}, mean = {labels.mean():.6f}")
        return feats, labels

    transform = make_transform()
    ds = UltrasoundDataset(root, transform)
    if len(ds) == 0:
        raise RuntimeError(f"Found 0 images under {root}.")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    encoder = build_encoder()
    feats, labels = extract_embeddings(encoder, loader)
    np.save(feat_path, feats)
    np.save(label_path, labels)
    print(f"DEBUG: Generated all_feats.shape = {feats.shape}, all_labels.shape = {labels.shape}")
    print(f"DEBUG: Generated all_labels min = {labels.min()}, max = {labels.max()}, mean = {labels.mean():.6f}")
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
# 3. Dynamic Score Thresholding & JAX Training Step
# ---------------------------------------------------------------------------
def init_params(key, dim):
    return {
        "w": jax.random.normal(key, (dim,)) * 0.01,
        "b": jnp.zeros(()),
    }


def get_prediction_thresholds(y_pred, num_thresholds=100):
    """Fixed-size (100,) thresholds, derived from the distribution of y_pred
    via quantiles rather than a fixed [0,1] grid. Shape stays constant
    regardless of len(y_pred), matching what pv_loss/compute_soft_roc expect."""
    eps = 1e-5
    q = jnp.linspace(1.0 - eps, eps, num_thresholds)
    thresholds = jnp.quantile(y_pred, q)
    return jax.lax.stop_gradient(thresholds)


def train_logreg(feats, labels, epochs=EPOCHS, lr=LR, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jnp.asarray(feats, dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.float32)

    params = init_params(key, x.shape[1])
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    P = jnp.sum(y == 1.0)
    N = jnp.sum(y == 0.0)
    kappa = 0.5 * (P + N)
    alpha = 0.6
    min_fp_cost_ratio = 1 / 9
    max_fp_cost_ratio = 1 / 6
    n_points = 1000
    temp = 0.03

    def loss_fn(p):
        logits = x @ p["w"] + p["b"]
        y_pred = jax.nn.sigmoid(logits)

        # Dynamic thresholds calculated directly from model output
        thresholds = get_prediction_thresholds(y_pred)

        return pv_loss(
            p, x, y, P, N, kappa, alpha, thresholds,
            min_fp_cost_ratio, max_fp_cost_ratio, n_points, temp
        )

    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  pvoros_loss {float(loss):.4f}")

    return params


# ---------------------------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = get_or_build_embeddings(DATA_DIR)
    print(f"DEBUG: After get_or_build_embeddings: all_feats.shape = {all_feats.shape}, all_labels.shape = {all_labels.shape}")
    print(f"DEBUG: After get_or_build_embeddings: all_labels min = {all_labels.min()}, max = {all_labels.max()}, mean = {all_labels.mean():.6f}")

    # Corrected assignment for train_test_split output
    train_feats, val_feats, train_labels, val_labels = split_train_val(all_feats, all_labels)

    print(f"DEBUG: After split_train_val:")
    print(f"DEBUG:   train_feats.shape = {train_feats.shape}, val_feats.shape = {val_feats.shape}")
    print(f"DEBUG:   train_labels.shape = {train_labels.shape}, val_labels.shape = {val_labels.shape}")
    print(f"DEBUG:   train_labels min = {train_labels.min()}, max = {train_labels.max()}, mean = {train_labels.mean():.6f}")
    print(f"DEBUG:   val_labels min = {val_labels.min()}, max = {val_labels.max()}, mean = {val_labels.mean():.6f}")

    print(f"Train samples: {train_feats.shape[0]}, Malignant rate: {train_labels.mean():.3f}")
    print(f"Val samples:   {val_feats.shape[0]}, Malignant rate: {val_labels.mean():.3f}")

    params = train_logreg(train_feats, train_labels)

    # Validation Evaluation
    x_val = jnp.asarray(val_feats, dtype=jnp.float32)
    y_val = jnp.asarray(val_labels, dtype=jnp.float32)
    logits_val = x_val @ params["w"] + params["b"]
    y_pred_val = jax.nn.sigmoid(logits_val)
    thresholds_val = get_prediction_thresholds(y_pred_val)

    P_val = jnp.sum(y_val == 1.0)
    N_val = jnp.sum(y_val == 0.0)
    kappa_val = 0.5 * (P_val + N_val)

    val_loss = pv_loss(
        params, x_val, y_val, P_val, N_val, kappa_val, 0.6,
        thresholds_val, 1/9, 1/6, 1000, 0.03
    )
    print(f"Val PVOROS loss: {float(val_loss):.4f}")

    np.save(CACHE_DIR / "logreg_w.npy", np.asarray(params["w"]))
    np.save(CACHE_DIR / "logreg_b.npy", np.asarray(params["b"]))
    print("Saved trained head weights to cache.")


if __name__ == "__main__":
    main()