"""
find_best_k.py
──────────────
Optional helper – run BEFORE main.py if you are unsure how many clusters
to use.  Plots the K-Means inertia (elbow method) and silhouette scores
for k = 2 … MAX_K so you can pick the best N_CLUSTERS for main.py.

Usage:
    python find_best_k.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_extraction import load_features

OUTPUT_DIR = "output"
MAX_K      = 15


def main():
    print("Loading cached features …")
    paths, features = load_features(OUTPUT_DIR)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_components = min(50, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    inertias   = []
    silhouettes = []
    ks = range(2, min(MAX_K + 1, len(paths)))

    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=42, max_iter=300)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, labels,
                               sample_size=min(500, len(paths)))
        silhouettes.append(sil)
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="grey")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    ax1.plot(list(ks), inertias, "o-", color="#58a6ff")
    ax1.set_title("Elbow – Inertia", color="white")
    ax1.set_xlabel("k", color="grey")
    ax1.set_ylabel("Inertia", color="grey")

    ax2.plot(list(ks), silhouettes, "s-", color="#7ee787")
    ax2.set_title("Silhouette Score", color="white")
    ax2.set_xlabel("k", color="grey")
    ax2.set_ylabel("Score", color="grey")

    out = Path(OUTPUT_DIR) / "elbow_plot.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out}")
    print("Pick the k where inertia stops dropping steeply AND silhouette is high.")


if __name__ == "__main__":
    main()
