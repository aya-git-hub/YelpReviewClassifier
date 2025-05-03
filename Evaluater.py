#!/usr/bin/env python3
"""
evaluate_clustering.py

Load two encoded datasets (HDF5 and JSON-lines), reduce dimensionality via PCA,
perform MiniBatchKMeans clustering with progress reporting, compute silhouette scores
on the reduced datasets, and display results—without command-line arguments.
"""

import h5py
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# --- Data Loading Functions ---

def load_hdf5(path: str) -> np.ndarray:
    """
    Load an array from an HDF5 file. Assumes the first dataset is the target array.
    """
    with h5py.File(path, 'r') as f:
        dataset_name = list(f.keys())[0]
        data = f[dataset_name][:]
    return data


def load_json_sequences(path: str) -> np.ndarray:
    """
    Load integer-encoded "text" sequences from a JSON-lines file.
    Each line: JSON object with a "text" list of ints.
    Returns a 2D NumPy array of shape (n_samples, max_sequence_length),
    padding shorter sequences with zeros.
    """
    sequences = []
    max_len = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            seq = record.get('text', [])
            sequences.append(seq)
            max_len = max(max_len, len(seq))
    arr = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        arr[i, :len(seq)] = seq
    return arr


# --- Dimensionality Reduction ---

def reduce_dimensionality(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Reduce X to n_components via PCA, print explained variance.
    """
    print(f"Reducing dimensionality from {X.shape[1]} to {n_components} components via PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    ev = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance ratio sum: {ev:.4f}\n")
    return X_reduced


# --- Clustering and Evaluation ---

def cluster_and_evaluate(X: np.ndarray, n_clusters: int, label: str) -> float:
    """
    Cluster X via MiniBatchKMeans (verbose output), then compute silhouette score.
    Prints summary and returns the score.
    """
    batch_size = min(10000, X.shape[0])
    print(f"Clustering {label} with MiniBatchKMeans (k={n_clusters}, batch_size={batch_size})...")
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        max_iter=100,
        init='k-means++',
        verbose=1
    )
    labels = mbk.fit_predict(X)
    print(f"Finished clustering {label} (inertia={mbk.inertia_:.2f}). Computing silhouette score on reduced data...")
    score = silhouette_score(X, labels)
    print(f"Dataset: {label} (reduced)")
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Silhouette Score: {score:.4f}\n")
    return score


def main():
    # Hard-coded paths and parameters
    full_data_path     = "dataset/data.h5"
    filtered_json_path = "dataset/non_useless_reviews_enc.json"
    n_clusters         = 5
    pca_components     = 50  # reduce to 50 dimensions

    print(f"Loading full dataset from: {full_data_path}")
    full_data = load_hdf5(full_data_path)
    print(f"Loading filtered dataset from: {filtered_json_path}")
    filtered_data = load_json_sequences(filtered_json_path)

    # Dimensionality reduction
    full_reduced = reduce_dimensionality(full_data, pca_components)
    filt_reduced = reduce_dimensionality(filtered_data, pca_components)

    print("\nRunning clustering and evaluation...\n" + "-"*50)
    score_full = cluster_and_evaluate(full_reduced, n_clusters, "Full Dataset")
    score_filt = cluster_and_evaluate(filt_reduced, n_clusters, "Filtered Dataset")

    print("Comparison:")
    print(f"  Full dataset silhouette score:     {score_full:.4f}")
    print(f"  Filtered dataset silhouette score: {score_filt:.4f}")


if __name__ == '__main__':
    main()
