"""
feature_extraction.py
─────────────────────
Step 2 of the VisionSort AI pipeline.

Builds a compact, meaningful numerical descriptor for every preprocessed image
by fusing three complementary feature types:

  1. Colour Histogram  – captures the global colour distribution (HSV space,
                         64 bins per channel → 192 values)
  2. ORB Keypoints     – local texture / structure descriptors aggregated into
                         a Bag-of-Visual-Words–style mean vector (32-D)
  3. Spatial Pyramid   – 4×4 grid of average pixel values on a resized 64×64
                         greyscale thumbnail → 1 024 values (captures layout)

All three vectors are L2-normalised and concatenated, then saved to
<output_dir>/features.npz so the clustering stage can reload them quickly.

Returns
-------
image_paths    : list[str]   – ordered list of image file paths
feature_matrix : np.ndarray  – shape (N, D) where N=images, D=feature dims
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Thumbnail size used for the spatial-pyramid descriptor
THUMB_SIZE = 64
# Number of ORB keypoints to detect per image
ORB_N_FEATURES = 500
# Colour histogram bins per HSV channel
HIST_BINS = 64


# ─── helper: colour histogram ─────────────────────────────────────────────────

def _colour_histogram(img: np.ndarray) -> np.ndarray:
    """
    Compute a normalised HSV colour histogram.
    Using HSV is more robust to lighting changes than RGB.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hists = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [HIST_BINS], [0, 256])
        hists.append(hist.flatten())
    vec = np.concatenate(hists).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-7)


# ─── helper: ORB mean descriptor ─────────────────────────────────────────────

_orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)

def _orb_mean(img: np.ndarray) -> np.ndarray:
    """
    Detect ORB keypoints, average their 32-byte descriptors into a single
    32-D vector.  Returns zeros if no keypoints are found.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, descs = _orb.detectAndCompute(grey, None)
    if descs is None or len(descs) == 0:
        return np.zeros(32, dtype=np.float32)
    vec = descs.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-7)


# ─── helper: spatial pyramid (4×4 grid) ──────────────────────────────────────

def _spatial_pyramid(img: np.ndarray) -> np.ndarray:
    """
    Resize to THUMB_SIZE × THUMB_SIZE, split into a 4×4 grid,
    take the mean pixel value of each cell.
    Encodes rough layout information that colour histograms miss.
    """
    grey  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(grey, (THUMB_SIZE, THUMB_SIZE),
                       interpolation=cv2.INTER_AREA)
    cells = []
    step  = THUMB_SIZE // 4
    for r in range(4):
        for c in range(4):
            cell = thumb[r*step:(r+1)*step, c*step:(c+1)*step]
            cells.append(cell.mean())
    vec = np.array(cells, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-7)


# ─── public API ──────────────────────────────────────────────────────────────

def extract_features(image_paths: list[str],
                     output_root: str) -> tuple[list[str], np.ndarray]:
    """
    Build the feature matrix for a list of preprocessed image paths.

    Parameters
    ----------
    image_paths  : ordered list of image file paths
    output_root  : root output folder (features saved here)

    Returns
    -------
    (image_paths, feature_matrix)  – feature_matrix shape: (N, D)
    """
    cache_path = Path(output_root) / "features.npz"

    vectors    = []
    valid_paths = []

    for path in tqdm(image_paths, desc="Extracting features", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [warn] Skipping unreadable image: {path}")
            continue

        # Concatenate all three descriptor families
        feat = np.concatenate([
            _colour_histogram(img),   # 192-D  (colour)
            _orb_mean(img),           #  32-D  (texture / structure)
            _spatial_pyramid(img),    #  16-D  (layout)
        ])

        vectors.append(feat)
        valid_paths.append(path)

    feature_matrix = np.vstack(vectors).astype(np.float32)

    # Persist to disk so you can re-run clustering without re-extracting
    np.savez_compressed(str(cache_path),
                        paths=np.array(valid_paths),
                        features=feature_matrix)
    print(f"  Features saved → {cache_path}")

    return valid_paths, feature_matrix


def load_features(output_root: str) -> tuple[list[str], np.ndarray]:
    """
    Reload features saved by a previous run (avoids re-extraction).
    """
    cache_path = Path(output_root) / "features.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached features at {cache_path}. Run extract_features first.")
    data = np.load(str(cache_path), allow_pickle=True)
    return list(data["paths"]), data["features"]
