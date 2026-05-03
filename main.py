"""
VisionSort AI - Main Pipeline
Run this file to process your entire VisionSort folder.
Usage: python main.py
"""

import os
import time
from pathlib import Path

from preprocessing import preprocess_images
from feature_extraction import extract_features
from clustering import cluster_images, visualize_clusters
from similarity_search import build_index, search_similar

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FOLDER   = "VisionSort"          # Folder with your raw images
OUTPUT_FOLDER  = "output"              # Where results are saved
N_CLUSTERS     = 7                    # Number of clusters (tune as needed)
N_SIMILAR      = 5                     # Top-N similar images to retrieve
QUERY_IMAGE    = None                  # Set to an image path for similarity search
                                       # e.g. "VisionSort/Screenshot_2026-04-06_162248.png"
# ───────────────────────────────────────────────────────────────────────────────


def main():
    start = time.time()

    print("=" * 60)
    print("  VisionSort AI: Automated Image Understanding & Organization")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/4] Preprocessing images …")
    processed = preprocess_images(INPUT_FOLDER, OUTPUT_FOLDER)
    print(f"      ✓ {len(processed)} images preprocessed")

    # 2. Feature Extraction
    print("\n[2/4] Extracting features …")
    image_paths, feature_matrix = extract_features(processed, OUTPUT_FOLDER)
    print(f"      ✓ Feature matrix shape: {feature_matrix.shape}")

    # 3. Clustering
    print("\n[3/4] Clustering images …")
    labels, reduced = cluster_images(image_paths, feature_matrix,
                                     n_clusters=N_CLUSTERS,
                                     output_dir=OUTPUT_FOLDER)
    visualize_clusters(image_paths, labels, reduced, OUTPUT_FOLDER)
    print(f"      ✓ Grouped into {N_CLUSTERS} clusters")

    # 4. Similarity Search
    print("\n[4/4] Building similarity index …")
    index = build_index(image_paths, feature_matrix)

    query = QUERY_IMAGE or image_paths[0]   # default: first image
    results = search_similar(query, image_paths, feature_matrix, index,
                             top_n=N_SIMILAR, output_dir=OUTPUT_FOLDER)
    print(f"      ✓ Similarity search done for: {Path(query).name}")
    print(f"        Top-{N_SIMILAR} matches:")
    for rank, (path, dist) in enumerate(results, 1):
        print(f"          {rank}. {Path(path).name}  (dist={dist:.4f})")

    elapsed = time.time() - start
    print(f"\n✅  Pipeline complete in {elapsed:.1f}s")
    print(f"    Results saved to: {OUTPUT_FOLDER}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
