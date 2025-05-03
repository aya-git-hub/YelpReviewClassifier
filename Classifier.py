#!/usr/bin/env python3
"""
evaluate_clustering.py

Load two encoded HDF5 datasets, apply PCA to reduce to 50 dimensions,
sample 10% of each reduced dataset, perform KMeans clustering on each sample,
and compute silhouette scores.
"""

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_hdf5(path: str) -> np.ndarray:
    """
    Load the first dataset from an HDF5 file.
    """
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        return f[key][:]

def reduce_pca(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Reduce X to n_components via PCA and print explained variance.
    """
    print(f"Reducing data from {X.shape[1]} to {n_components} dimensions via PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_red = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance ratio sum: {explained:.4f}\n")
    return X_red

def sample_data(X: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """
    Randomly sample a fraction of rows from X.
    """
    n = X.shape[0]
    k = max(1, int(n * fraction))
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=k, replace=False)
    print(f"Sampling {k} of {n} rows ({fraction*100:.1f}% fraction)\n")
    return X[idx]

def cluster_and_score(X: np.ndarray, n_clusters: int = 5) -> float:
    """
    Cluster X with KMeans and return silhouette score.
    """
    print(f"Clustering sample with k={n_clusters} clusters...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    return score

def main():
    full_path     = "dataset/data.h5"
    filtered_path = "dataset/data_non_u.h5"
    n_clusters    = 5

    print(f"Loading full dataset: {full_path}")
    full_data = load_hdf5(full_path)
    print(f"Loading filtered dataset: {filtered_path}\n")
    filt_data = load_hdf5(filtered_path)

    # PCA reduction
    full_pca = reduce_pca(full_data, 50)
    filt_pca = reduce_pca(filt_data, 50)

    # sampling
    full_samp = sample_data(full_pca, 0.08)
    filt_samp = sample_data(filt_pca, 0.08)

    # clustering & scoring
    sil_full = cluster_and_score(full_samp, n_clusters)
    print(f"Full dataset sample silhouette: {sil_full:.4f}\n")

    sil_filt = cluster_and_score(filt_samp, n_clusters)
    print(f"Filtered dataset sample silhouette: {sil_filt:.4f}\n")

if __name__ == '__main__':
    main()
