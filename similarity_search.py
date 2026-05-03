"""
similarity_search.py
────────────────────
Step 4 of the VisionSort AI pipeline.

Given ONE query image, returns the most visually similar images from the dataset.

Method
------
• Features are already L2-normalised during extraction, so cosine distance
  reduces to a simple dot product: similarity = featureA · featureB.
• An exact nearest-neighbour search is used (scikit-learn BallTree with
  Euclidean metric on the normalised vectors, which is equivalent to cosine).
• For datasets > 5 000 images you can swap BallTree for faiss (commented below).

Outputs
-------
• <output_dir>/similarity_<query_stem>.png  – contact sheet of query + top-N
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.neighbors import BallTree


THUMB_W, THUMB_H = 200, 150   # thumbnail size in the output contact sheet


# ─── helpers ──────────────────────────────────────────────────────────────────

def _thumb(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def _contact_sheet(query_path: str,
                   results: list[tuple[str, float]],
                   output_dir: str):
    """Save a matplotlib figure with query on left, similar images on right."""
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3, 4))
    fig.patch.set_facecolor("#0d1117")

    # Query image
    axes[0].imshow(_thumb(query_path))
    axes[0].set_title("Query", color="#58a6ff", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    # Draw a coloured border
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#58a6ff")
        spine.set_linewidth(3)

    # Similar images
    for i, (path, dist) in enumerate(results):
        ax = axes[i + 1]
        ax.imshow(_thumb(path))
        ax.set_title(f"#{i+1}  d={dist:.3f}",
                     color="#7ee787", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"Similarity Search — {Path(query_path).name}",
                 color="white", fontsize=11, y=1.02)
    plt.tight_layout()

    out = Path(output_dir) / f"similarity_{Path(query_path).stem}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Contact sheet saved → {out}")


# ─── public API ──────────────────────────────────────────────────────────────

def build_index(image_paths: list[str],
                feature_matrix: np.ndarray):
    """
    Build a BallTree index over the feature matrix.

    For datasets of ~100 images this is instantaneous; BallTree scales well
    to tens of thousands.  Replace with faiss.IndexFlatIP for millions.

    Returns the fitted BallTree object.
    """
    # BallTree expects float64
    tree = BallTree(feature_matrix.astype(np.float64), leaf_size=20,
                    metric="euclidean")
    return tree


def search_similar(query_path: str,
                   image_paths: list[str],
                   feature_matrix: np.ndarray,
                   index,                       # BallTree
                   top_n: int = 5,
                   output_dir: str = "output") -> list[tuple[str, float]]:
    """
    Find the *top_n* most similar images to *query_path*.

    The query image is looked up in *image_paths*; if it's not there (e.g. it
    is an external image) its features are extracted on the fly.

    Returns
    -------
    List of (image_path, distance) tuples, sorted ascending by distance.
    The query image itself is excluded from results.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Get query feature vector ─────────────────────────────────────────────
    try:
        q_idx = image_paths.index(query_path)
        q_vec = feature_matrix[q_idx].reshape(1, -1)
    except ValueError:
        # Query is an external image → extract features on the fly
        from feature_extraction import (
            _colour_histogram, _orb_mean, _spatial_pyramid
        )
        img = cv2.imread(str(query_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read query image: {query_path}")
        q_vec = np.concatenate([
            _colour_histogram(img),
            _orb_mean(img),
            _spatial_pyramid(img),
        ]).reshape(1, -1).astype(np.float64)
        q_idx = -1   # mark as external

    # ── Nearest-neighbour query ───────────────────────────────────────────────
    k = top_n + 1    # +1 to account for the query itself (if present)
    k = min(k, len(image_paths))

    distances, indices = index.query(q_vec.astype(np.float64), k=k)
    distances = distances[0]
    indices   = indices[0]

    results = []
    for dist, idx in zip(distances, indices):
        if idx == q_idx:      # skip the query image
            continue
        results.append((image_paths[idx], float(dist)))
        if len(results) == top_n:
            break

    # ── Visual output ────────────────────────────────────────────────────────
    _contact_sheet(query_path, results, output_dir)

    return results
