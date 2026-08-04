import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ensure metrics_jax is available for pvoros_score
from metrics_jax import pvoros_score

# -------------------------------------------------------------------------
# 1. Dataset setup for BUSI (Excluding Mask Files)
# -------------------------------------------------------------------------
LABEL_MAP = {"benign": 0, "normal": 0, "malignant": 1}

class BUSIDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.paths, self.labels = [], []
        
        for cls, label in LABEL_MAP.items():
            cls_folder = root / cls
            if not cls_folder.exists():
                continue
                
            for p in cls_folder.glob("*"):
                # Exclude mask files, keep original images
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "_mask" not in p.stem:
                    self.paths.append(p)
                    self.labels.append(label)
                    
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ViT default transform with scaling/cropping as described in paper
weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

# -------------------------------------------------------------------------
# 2. Extract Features using default PyTorch ViT
# -------------------------------------------------------------------------
def extract_vit_features(data_loader, model, device):
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            # Pass directly through ViT model to obtain output representations
            out = model(imgs) 
            features.append(out.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = vit_b_16(weights=weights).to(device)

data_dir = Path("busi_data") 
dataset = BUSIDataset(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

X, y = extract_vit_features(loader, vit_model, device)

# -------------------------------------------------------------------------
# 3. Stratified 5-Fold Cross-Validation Evaluation
# -------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf = LogisticRegression(C=1000, max_iter=25, solver='lbfgs', random_state=42)
        clf.fit(X_train_scaled, y_train)
        
    oof_preds[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]

# -------------------------------------------------------------------------
# 4. Metrics Computation
# -------------------------------------------------------------------------
auroc = roc_auc_score(y, oof_preds)
voros = pvoros_score(y, oof_preds, 0.00001, 1.0, 0, 1)

print(f"AUROC: {auroc * 100:.1f}%")
print(f"VOROS: {float(voros) * 100:.1f}%")