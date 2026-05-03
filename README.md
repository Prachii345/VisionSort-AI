<div align="center">

# 🔍 VisionSort AI

### Automated Image Understanding and Organization System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **VisionSort AI** takes a folder of random, unlabelled images and automatically cleans, understands, groups, and searches through them — with **zero manual labelling required.**

<br/>

---

</div>

## 📌 What Does It Do?

Given a folder of **any random images**, VisionSort AI will:

- 🧹 **Clean** every image (remove noise, enhance contrast)
- 🔢 **Understand** every image (convert to 240-D numerical vectors)
- 🤖 **Group** similar images together automatically (unsupervised clustering)
- 🔎 **Find** the most visually similar images to any query image

All in under **30 seconds** on a standard CPU — no GPU, no labels, no manual work.

---

## ✨ Features at a Glance

| Feature | Description |
|---|---|
| 🧹 Preprocessing | Gaussian Blur denoising + CLAHE contrast enhancement |
| 🔢 Feature Extraction | Colour Histograms + ORB Keypoints + Spatial Pyramid → 240-D vector |
| 🤖 Clustering | PCA dimensionality reduction + K-Means++ grouping |
| 🔎 Similarity Search | BallTree nearest-neighbour search on L2-normalised vectors |
| 📊 Visualizations | Cluster grids, PCA scatter plots, elbow plots, contact sheets |

---

## ⚙️ How The Pipeline Works

```
📁 Raw Images (VisionSort/)
         │
         ▼
┌─────────────────────────────────────┐
│  STEP 1 — preprocessing.py          │
│  • Gaussian Blur  → noise removal   │
│  • CLAHE on LAB   → contrast boost  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  STEP 2 — feature_extraction.py     │
│  • Colour Histogram  →  192-D       │
│  • ORB Mean Descriptor →  32-D      │
│  • Spatial Pyramid 4×4 →  16-D      │
│  ─────────────────────────────────  │
│  Total Feature Vector  →  240-D     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  STEP 3 — clustering.py             │
│  • StandardScaler → normalise       │
│  • PCA → 50 components (98.7% var)  │
│  • K-Means++ → group into clusters  │
│  • Silhouette Score → evaluate      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  STEP 4 — similarity_search.py      │
│  • BallTree index on feature matrix │
│  • Query → Top-N similar images     │
│  • Contact sheet visual output      │
└─────────────────────────────────────┘
         │
         ▼
📂 output/  (results saved here)
```

---

## 🗂️ Project Structure

```
VisionSortAI/
│
├── 📁 VisionSort/                  ← Put your input images here
│
├── 🐍 main.py                      ← Run this to start everything
├── 🐍 preprocessing.py             ← Noise removal + contrast enhancement
├── 🐍 feature_extraction.py        ← Converts images to 240-D vectors
├── 🐍 clustering.py                ← PCA + K-Means clustering
├── 🐍 similarity_search.py         ← Find visually similar images
├── 🐍 find_best_k.py               ← Elbow method to find best k
├── 📄 requirements.txt             ← Python dependencies
│
└── 📁 output/                      ← Auto-created after running
    ├── 📁 preprocessed/            ← Cleaned images
    ├── 📁 clusters/
    │   ├── 📁 cluster_00/          ← Images in cluster 0
    │   ├── 📁 cluster_01/          ← Images in cluster 1
    │   └── ...
    ├── 📊 cluster_grid.png         ← Visual thumbnail grid
    ├── 📊 pca_scatter.png          ← 2D scatter plot
    ├── 📊 elbow_plot.png           ← Best k analysis
    ├── 📊 similarity_*.png         ← Similarity search results
    ├── 📄 cluster_assignments.csv  ← Image → cluster label table
    └── 💾 features.npz             ← Cached feature matrix
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Prachii345/VisionSort-AI.git
cd VisionSort-AI
```

### 2️⃣ Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add your images
Place all your images inside the `VisionSort/` folder.
```
VisionSort/
├── image1.png
├── image2.jpg
└── ...
```

### 5️⃣ Run the full pipeline
```bash
python main.py
```

### 6️⃣ (Optional) Find the best number of clusters
```bash
python find_best_k.py
```
Open `output/elbow_plot.png` — pick the k where the curve bends, then update `main.py`.

---

## 🎛️ Configuration

Open `main.py` and adjust these at the top:

```python
INPUT_FOLDER = "VisionSort"    # 📁 Folder with your images
OUTPUT_FOLDER = "output"       # 📂 Where results are saved
N_CLUSTERS   = 7               # 🔢 Number of clusters
N_SIMILAR    = 5               # 🔎 Top-N similar images to find
QUERY_IMAGE  = None            # 🖼️ Set to an image path for similarity search
```

---

## 📊 Results on Sample Dataset (70 Images)

| Metric | Value |
|---|---|
| ✅ Images Processed | 70 |
| ✅ Feature Vector Size | 240-D |
| ✅ PCA Variance Retained | 98.7% |
| ✅ Number of Clusters | 7 |
| ✅ Silhouette Score | 0.069 |
| ✅ Pipeline Runtime | ~28 seconds |
| ✅ Similarity Distance Range | 0.587 – 0.728 |

> **Note on Silhouette Score:** A score of 0.069 is expected for a completely random, diverse dataset. A dataset of only cats/dogs/cars would score 0.5+.

---

## 📁 Output Examples

### 🔹 Cluster Grid (`cluster_grid.png`)
Each row = one cluster. Images in the same row are grouped by visual similarity.

### 🔹 PCA Scatter (`pca_scatter.png`)
Every image plotted in 2D space. Each colour = different cluster.

### 🔹 Elbow Plot (`elbow_plot.png`)
Inertia and Silhouette score vs k — use this to choose the best number of clusters.

### 🔹 Similarity Search (`similarity_*.png`)
Query image on the left, top-5 most visually similar images on the right with distance scores.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square) | 4.8+ | Image preprocessing, ORB features |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square) | 1.26 | Numerical computations |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square) | 1.3+ | PCA, K-Means, BallTree, Silhouette |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) | 3.7+ | Visualizations |
| ![tqdm](https://img.shields.io/badge/tqdm-FFC107?style=flat-square) | 4.65+ | Progress bars |

---

## 🔮 Future Work — Deep Learning Version

The classical approach groups images by **colour and texture**.
A deep learning upgrade using **pretrained ResNet50** would enable **semantic grouping** — grouping cats with cats, cars with cars.

| Aspect | Classical (Current) | Deep Learning (Future) |
|---|---|---|
| Features | Handcrafted 240-D | ResNet50 embeddings 2048-D |
| Semantic Understanding | ❌ No | ✅ Yes |
| Expected Silhouette | 0.07 | 0.25+ |
| GPU Required | ❌ No | ✅ Recommended |

> The deep learning module (`feature_extraction_deeplearning.py`) is already included — just swap the file and install PyTorch!

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify and distribute.

---

<div align="center">

## 👩‍💻 Author

**Prachii345**

[![GitHub](https://img.shields.io/badge/GitHub-Prachii345-181717?style=for-the-badge&logo=github)](https://github.com/Prachii345)

<br/>

*Built with ❤️ using Python and Computer Vision*

</div>
