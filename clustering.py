"""
clustering.py
─────────────
Step 3 of the VisionSort AI pipeline.

Two-stage dimensionality-reduction + clustering:

  Stage A – PCA
    Reduces the high-dimensional feature vector (240-D) down to
    `PCA_COMPONENTS` dimensions while retaining ≥ 95 % of variance.
    This speeds up K-Means and removes noisy/redundant dimensions.

  Stage B – K-Means
    Groups images into `n_clusters` clusters based on their PCA coordinates.
    Uses k-means++ initialisation for stable, reproducible results.

Outputs
-------
• <output_dir>/clusters/cluster_<k>/   – symlinks (or copies) of member images
• <output_dir>/cluster_assignments.csv – image → cluster label mapping
• <output_dir>/pca_scatter.png         – 2-D PCA scatter coloured by cluster
• <output_dir>/cluster_grid.png        – thumbnail grid per cluster
"""

import os
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering (works in VS Code terminal)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
from tqdm import tqdm

PCA_COMPONENTS   = 50    # Maximum PCA dimensions (auto-reduced if N < 50)
THUMB_W, THUMB_H = 120, 90   # Thumbnail size for the cluster grid


# ─── internal helpers ─────────────────────────────────────────────────────────

def _make_thumbnail(img_path: str) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    return cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def _save_cluster_folders(image_paths: list[str],
                           labels: np.ndarray,
                           output_root: str):
    """Copy images into per-cluster sub-folders."""
    clusters_root = Path(output_root) / "clusters"
    # Clean up from previous run
    if clusters_root.exists():
        shutil.rmtree(clusters_root)

    for label, path in zip(labels, image_paths):
        dest_dir = clusters_root / f"cluster_{label:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / Path(path).name)


def _save_csv(image_paths: list[str],
              labels: np.ndarray,
              output_root: str):
    csv_path = Path(output_root) / "cluster_assignments.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "cluster"])
        for path, lbl in zip(image_paths, labels):
            writer.writerow([Path(path).name, int(lbl)])
    print(f"  Cluster assignments saved → {csv_path}")


def _scatter_plot(reduced_2d: np.ndarray,
                  labels: np.ndarray,
                  n_clusters: int,
                  output_root: str,
                  sil_score: float):
    """2-D PCA scatter plot coloured by cluster label."""
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                   color=cmap(k), s=40, alpha=0.8, label=f"Cluster {k}")

    ax.set_title(f"PCA Scatter – {n_clusters} clusters  |  Silhouette={sil_score:.3f}",
                 color="white", fontsize=13)
    ax.set_xlabel("PC 1", color="grey")
    ax.set_ylabel("PC 2", color="grey")
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(loc="best", fontsize=8, framealpha=0.3,
              labelcolor="white", facecolor="#111")

    out = Path(output_root) / "pca_scatter.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  PCA scatter saved → {out}")


def _cluster_grid(image_paths: list[str],
                  labels: np.ndarray,
                  n_clusters: int,
                  output_root: str,
                  max_per_cluster: int = 8):
    """One row per cluster, columns = sample thumbnails."""
    rows = []
    for k in range(n_clusters):
        members = [p for p, l in zip(image_paths, labels) if l == k]
        sample  = members[:max_per_cluster]
        thumbs  = [_make_thumbnail(p) for p in sample]

        # Pad row to max_per_cluster columns
        while len(thumbs) < max_per_cluster:
            thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))

        # Label column
        label_col = np.zeros((THUMB_H, 80, 3), dtype=np.uint8)
        cv2.putText(label_col, f"C{k:02d}", (5, THUMB_H // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        row = np.hstack([label_col] + thumbs)
        rows.append(row)

    grid = np.vstack(rows)
    out  = Path(output_root) / "cluster_grid.png"
    cv2.imwrite(str(out), grid)
    print(f"  Cluster grid saved → {out}")


# ─── public API ──────────────────────────────────────────────────────────────

def cluster_images(image_paths: list[str],
                   feature_matrix: np.ndarray,
                   n_clusters: int = 8,
                   output_dir: str = "output") -> tuple[np.ndarray, np.ndarray]:
    """
    Run PCA + K-Means on *feature_matrix*.

    Returns
    -------
    labels    : np.ndarray shape (N,)  – cluster index for each image
    reduced   : np.ndarray shape (N, 2) – first two PCA components (for plotting)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_samples = feature_matrix.shape[0]

    # Guard: can't have more clusters than samples
    n_clusters = min(n_clusters, n_samples)

    # ── Step A: Standardise then PCA ────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    n_components = min(PCA_COMPONENTS, n_samples, feature_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA: {n_components} components explain "
          f"{explained*100:.1f}% of variance")

    # ── Step B: K-Means ──────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters,
                    init="k-means++",
                    n_init=15,          # run 15 times, keep best
                    max_iter=500,
                    random_state=42)
    labels = kmeans.fit_predict(X_pca)

    # Silhouette score (quality metric; higher = better separated clusters)
    if n_samples > n_clusters:
        sil = silhouette_score(X_pca, labels, sample_size=min(500, n_samples))
        print(f"  Silhouette score: {sil:.4f}  "
              f"(range −1 → 1; >0.2 is reasonable for real images)")
    else:
        sil = 0.0

    # ── Persist results ──────────────────────────────────────────────────────
    print("  Copying images into cluster folders …")
    _save_cluster_folders(image_paths, labels, output_dir)
    _save_csv(image_paths, labels, output_dir)

    return labels, X_pca[:, :2]   # return first 2 PCs for scatter


def visualize_clusters(image_paths: list[str],
                       labels: np.ndarray,
                       reduced_2d: np.ndarray,
                       output_dir: str):
    """Generate scatter plot and thumbnail grid after clustering."""
    n_clusters = int(labels.max()) + 1

    # Silhouette on reduced 2-D coords (approximate – fast)
    if len(image_paths) > n_clusters:
        sil = silhouette_score(reduced_2d, labels)
    else:
        sil = 0.0

    _scatter_plot(reduced_2d, labels, n_clusters, output_dir, sil)
    _cluster_grid(image_paths, labels, n_clusters, output_dir)
    print(f"  Visualisations written to {output_dir}/")
