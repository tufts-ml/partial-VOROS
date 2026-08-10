import os
import warnings
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

from jax import config
config.update("jax_enable_x64", True)

# Import VOROS metric functions from your JAX module
from metrics import voros_score


# -------------------------------------------------------------------------
# 1. Dataset Setup for BUSI (Excluding Mask Files)
# -------------------------------------------------------------------------
# Binary classification mapping: benign and normal = negative (0), malignant = positive (1)
LABEL_MAP = {"benign": 0, "normal": 0, "malignant": 1}

class BUSIDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.paths, self.labels = [], []
        
        for cls, label in LABEL_MAP.items():
            cls_folder = root / cls
            if not cls_folder.exists():
                continue
                
            for p in cls_folder.glob("*"):
                # Filter out mask image files (_mask.png) and keep original ultrasound scans
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "_mask" not in p.stem:
                    self.paths.append(p)
                    self.labels.append(label)
                    
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Convert image to RGB for PyTorch ViT standard input compatibility
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# Standard PyTorch ViT weights & default image preprocessing (Resize + CenterCrop to 224x224)
weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()


# -------------------------------------------------------------------------
# 2. Extract 768-Dim Feature Embeddings using PyTorch ViT
# -------------------------------------------------------------------------
def extract_vit_embeddings(data_loader, model, device):
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            
            # Forward pass through patch projection and transformer encoder layers
            x = model._process_input(imgs)
            n = x.shape[0]
            batch_class_token = model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = model.encoder(x)
            
            # Extract pooled [CLS] embedding vector (768 dimensions)
            cls_feats = x[:, 0]
            
            features.append(cls_feats.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


# -------------------------------------------------------------------------
# 3. Execution & Cross-Validated Evaluation
# -------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load Pretrained Vision Transformer
    vit_model = vit_b_16(weights=weights).to(device)

    # 2. Load Dataset
    data_dir = Path("busi_data")  # Update to your local dataset folder
    dataset = BUSIDataset(root=data_dir, transform=transform)
    print(f"Total ultrasound images loaded: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. Feature Extraction
    X, y = extract_vit_embeddings(loader, vit_model, device)
    print(f"Extracted feature shape: {X.shape}")

    # 4. Stratified 5-Fold Cross-Validation for Out-of-Fold Probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Logistic Regression matching paper specs: C=1000, max_iter=25
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf = LogisticRegression(C=1000, max_iter=25, solver='lbfgs', random_state=42)
            clf.fit(X_train_scaled, y_train)
            
        # Predict out-of-fold positive class probabilities
        oof_preds[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]

    # 5. Calculate Metrics
    auroc = roc_auc_score(y, oof_preds)
    print("\n--- Model Evaluation Results ---")
    print(f"AUROC: {auroc * 100:.1f}%")
    
    # Compute full VOROS score using JAX implementation
    voros = voros_score(y, oof_preds, 1e-8, 1)


    print(f"VOROS: {float(voros) * 100:.1f}%")

if __name__ == "__main__":
    main()