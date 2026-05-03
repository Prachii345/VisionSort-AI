"""
preprocessing.py
────────────────
Step 1 of the VisionSort AI pipeline.

Applies two classical image-enhancement steps to every image in the input folder:
  • Gaussian Blur   – reduces high-frequency noise
  • CLAHE           – contrast-limited adaptive histogram equalisation
                      (works per-channel in LAB colour space so colours stay natural)

Preprocessed images are written to <output_dir>/preprocessed/.
Returns a list of those output paths for the next stage.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _load_image(path: str) -> np.ndarray | None:
    """Read an image; return None on failure."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [warn] Could not read: {path}")
    return img


def _gaussian_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Gaussian blur to suppress sensor / compression noise.
    ksize=3 is light enough to preserve detail while removing speckles.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)


def _clahe_enhance(img: np.ndarray,
                   clip_limit: float = 2.0,
                   tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation.
    Converts to LAB, enhances only the L (lightness) channel, converts back.
    This avoids colour shifts that plain HE introduces.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_chan)

    enhanced_lab = cv2.merge([l_enhanced, a_chan, b_chan])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess_images(input_dir: str, output_root: str) -> list[str]:
    """
    Process every supported image in *input_dir*.

    Parameters
    ----------
    input_dir   : folder containing raw images (e.g. "VisionSort")
    output_root : root output folder (e.g. "output")

    Returns
    -------
    List of absolute paths to the preprocessed images.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_root) / "preprocessed"
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all images (non-recursive; add rglob for sub-folders)
    image_files = sorted([
        p for p in input_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        raise FileNotFoundError(
            f"No supported images found in '{input_dir}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    processed_paths = []

    for img_path in tqdm(image_files, desc="Preprocessing", unit="img"):
        img = _load_image(img_path)
        if img is None:
            continue

        # Pipeline: denoise → enhance contrast
        denoised = _gaussian_denoise(img)
        enhanced = _clahe_enhance(denoised)

        # Save with the same filename into the output folder
        out_file = output_path / img_path.name
        cv2.imwrite(str(out_file), enhanced)
        processed_paths.append(str(out_file))

    print(f"  Preprocessed images → {output_path}")
    return processed_paths
