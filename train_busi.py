"""
Ultrasound classifier: frozen ViT encoder -> logistic regression -> Partial VOROS objective.

Pipeline
--------
1. (PyTorch) Load a pretrained torchvision ViT-B/16, strip its classification head,
   and use it as a frozen feature encoder. Images are resized to 224x224 and
   replicated to 3 channels (ViT expects RGB, ultrasound images are grayscale).
2. Run every image through the encoder ONCE to get a 768-d embedding, and cache
   embeddings + labels to disk. No need to keep the ViT in the loop after this,
   since it's frozen -> this makes the actual training step trivial.
3. Split the cached embeddings into train/val with a stratified split (data
   is not pre-split on disk -- it's just DATA_DIR/{benign,malignant,normal}/*).
4. (JAX) Train a single linear layer (logistic regression) on the train split,
   with Partial VOROS as the loss via jax.grad.

Classes are collapsed to binary per your spec: malignant = positive (1),
benign + normal = negative (0).

Plug in your own partial-VOROS package
---------------------------------------
This script imports `partial_voros` and calls `partial_voros_loss(...)` inside
`pvoros_loss_fn`. That name/signature is a guess at your packaged API -- swap
in whatever you actually exposed (e.g. `from partial_voros.metric import ...`).
If the import fails, the script falls back to a small self-contained
differentiable Partial VOROS implemented the same way you described in your
own work: sigmoid-relaxed soft ROC points + shoelace-formula area, restricted
to a cost-ratio interval [c_lo, c_hi]. That fallback is here so the script is
runnable end-to-end even before you wire in the real library -- swap it out.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

import jax
import jax.numpy as jnp
import optax

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("busi_data")       # expects DATA_DIR/{benign,malignant,normal}/*.png -- not pre-split
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
VAL_FRACTION = 0.2
SPLIT_SEED = 0

# Partial VOROS cost-ratio interval to integrate over. Narrow this to the
# region of the cost space that's clinically relevant (e.g. cases where
# missing a malignancy is weighted much more heavily than a false alarm).
C_LO, C_HI = 0.1, 0.9

LR = 1e-2
EPOCHS = 300


# ---------------------------------------------------------------------------
# 1. Dataset — binary labels, benign+normal -> 0, malignant -> 1
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
        img = Image.open(self.paths[idx]).convert("L").convert("RGB")  # gray -> 3ch
        return self.transform(img), self.labels[idx]


def make_transform():
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    # Use the exact preprocessing the pretrained ViT was trained with
    # (resize to 224, ImageNet normalization) rather than hand-rolling it.
    return weights.transforms()


# ---------------------------------------------------------------------------
# 2. Frozen ViT encoder -> cache embeddings
# ---------------------------------------------------------------------------
def build_encoder():
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()   # drop classification head, keep 768-d CLS embedding
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


@torch.no_grad()
def extract_embeddings(model, loader):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        emb = model(x)                       # (B, 768)
        feats.append(emb.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


def get_or_build_embeddings(root: Path):
    """
    Encode the FULL dataset once (data isn't pre-split into train/val), cache
    it, and return everything. Splitting happens afterward in memory via
    `split_train_val`, so the ViT never needs to run twice.
    """
    feat_path = CACHE_DIR / "all_feats.npy"
    label_path = CACHE_DIR / "all_labels.npy"
    if feat_path.exists() and label_path.exists():
        return np.load(feat_path), np.load(label_path)

    transform = make_transform()
    ds = UltrasoundDataset(root, transform)
    if len(ds) == 0:
        raise RuntimeError(
            f"Found 0 images under {root}. Expected subfolders "
            f"{list(LABEL_MAP.keys())} directly under it, e.g. {root}/benign/*.png. "
            f"Check DATA_DIR and folder names."
        )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    encoder = build_encoder()
    feats, labels = extract_embeddings(encoder, loader)
    np.save(feat_path, feats)
    np.save(label_path, labels)
    return feats, labels


def split_train_val(feats, labels, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Stratified train/val split so both splits keep the same malignant rate."""
    from sklearn.model_selection import train_test_split

    train_feats, val_feats, train_labels, val_labels = train_test_split(
        feats, labels,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    return train_feats, train_labels, val_feats, val_labels


# ---------------------------------------------------------------------------
# 3. Partial VOROS loss (JAX) — swap in your real package here
# ---------------------------------------------------------------------------
try:
    import metrics_jax  # noqa: F401  -- your pip-editable package

    def pvoros_loss_fn(scores, labels, c_lo=C_LO, c_hi=C_HI):
        """Adjust this call to match your actual packaged API."""
        return metrics_jax.pvoros_loss(scores, labels, c_lo=c_lo, c_hi=c_hi)

    print("Using metrics_jax package for the training objective.")

except ImportError:
    print("`metrics_jax` not importable — using in-file differentiable fallback. "
          "Swap `pvoros_loss_fn` for your real package once it's on the path.")

    def _soft_roc_points(scores, labels, taus, sharpness=50.0):
        """Sigmoid-relaxed TPR/FPR at each threshold in `taus`."""
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = jnp.sum(pos_mask)
        n_neg = jnp.sum(neg_mask)

        # soft "predicted positive" indicator per threshold: sigmoid(k*(score - tau))
        pred_pos = jax.nn.sigmoid(sharpness * (scores[None, :] - taus[:, None]))  # (T, N)

        tpr = jnp.sum(pred_pos * pos_mask[None, :], axis=1) / jnp.maximum(n_pos, 1e-8)
        fpr = jnp.sum(pred_pos * neg_mask[None, :], axis=1) / jnp.maximum(n_neg, 1e-8)
        return fpr, tpr

    def _shoelace_area(x, y):
        """Shoelace formula for area under a polyline traced from (x,y) points."""
        x2 = jnp.concatenate([x, x[:1]])
        y2 = jnp.concatenate([y, y[:1]])
        return 0.5 * jnp.abs(jnp.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))

    def pvoros_loss_fn(scores, labels, c_lo=C_LO, c_hi=C_HI, n_taus=50):
        """
        Differentiable proxy for Partial VOROS: soft ROC curve, restricted to
        the FPR band implied by cost ratios [c_lo, c_hi] (approximated here as
        the same-numbered band on the FPR axis — replace with your exact
        cost-to-FPR mapping / convex-hull logic for a faithful metric).
        Returns a LOSS (lower is better), i.e. negative partial volume.
        """
        taus = jnp.linspace(0.0, 1.0, n_taus)
        scores = jax.nn.sigmoid(scores)  # squash logits to [0,1] to compare with taus
        fpr, tpr = _soft_roc_points(scores, labels, taus)

        band = (fpr >= c_lo) & (fpr <= c_hi)
        band = band.astype(jnp.float32)
        fpr_band = fpr * band
        tpr_band = tpr * band

        # sort by fpr ascending for a well-formed polygon
        order = jnp.argsort(fpr_band)
        fpr_sorted = fpr_band[order]
        tpr_sorted = tpr_band[order]

        area = _shoelace_area(fpr_sorted, tpr_sorted)
        return -area  # maximize area under partial ROC band -> minimize negative area


# ---------------------------------------------------------------------------
# 4. Logistic regression head, trained with PVOROS via jax.grad
# ---------------------------------------------------------------------------
def init_params(key, dim):
    w = jax.random.normal(key, (dim,)) * 0.01
    b = jnp.zeros(())
    return {"w": w, "b": b}


def forward(params, x):
    return x @ params["w"] + params["b"]   # logits


def loss_fn(params, x, y):
    logits = forward(params, x)
    return pvoros_loss_fn(logits, y)


@jax.jit
def train_step(params, opt_state, x, y, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_logreg(feats, labels, epochs=EPOCHS, lr=LR, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jnp.asarray(feats, dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.float32)

    params = init_params(key, x.shape[1])
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, x, y, optimizer)
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  pvoros_loss {loss:.4f}")

    return params


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
def main():
    all_feats, all_labels = get_or_build_embeddings(DATA_DIR)
    train_feats, train_labels, val_feats, val_labels = split_train_val(all_feats, all_labels)

    print(f"train: {train_feats.shape}, malignant rate {train_labels.mean():.3f}")
    print(f"val:   {val_feats.shape}, malignant rate {val_labels.mean():.3f}")

    params = train_logreg(train_feats, train_labels)

    val_logits = forward(params, jnp.asarray(val_feats, dtype=jnp.float32))
    val_probs = jax.nn.sigmoid(val_logits)
    val_loss = pvoros_loss_fn(val_logits, jnp.asarray(val_labels, dtype=jnp.float32))
    print(f"val PVOROS loss: {val_loss:.4f}")

    np.save(CACHE_DIR / "logreg_w.npy", np.asarray(params["w"]))
    np.save(CACHE_DIR / "logreg_b.npy", np.asarray(params["b"]))
    print("Saved weights to cache/logreg_w.npy, cache/logreg_b.npy")


if __name__ == "__main__":
    main()