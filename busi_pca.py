import numpy as np
from sklearn.decomposition import PCA
from train_busi import get_or_build_embeddings
from pathlib import Path

DATA_DIR = Path("busi_data")

# 1. Load your ViT-B/16 embeddings (typically shape [N, 768])
embeddings, labels = get_or_build_embeddings(DATA_DIR)
print(f"Original embeddings shape: {embeddings.shape}")

# 2. Fit PCA models for 2, 10, and 20 dimensions
dimensions = [2, 10, 20]
reduced_features = {}

for dim in dimensions:
    pca = PCA(n_components=dim)
    # Fit and transform the embeddings
    reduced_feats = pca.fit_transform(embeddings)
    reduced_features[dim] = reduced_feats
    
    # Check retained variance
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA ({dim:2d}D) shape: {reduced_feats.shape} | Retained Variance: {explained_var:.2f}%")
    
    # Save to separate files if needed
    np.save(f"feats_{dim}d.npy", reduced_feats)

# Access individual arrays:
# feats_2d  = reduced_features[2]
# feats_10d = reduced_features[10]
# feats_20d = reduced_features[20]