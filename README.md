# VisionSort-AI
Automated Image Understanding and Organization System

VisionSort AI is a fully modular Computer Vision pipeline that automatically preprocesses, understands, clusters, and searches through a collection of images — with zero manual labelling required.

✨ Features

🧹 Image Preprocessing — Gaussian blur denoising + CLAHE contrast enhancement
🔢 Feature Extraction — Colour histograms + ORB keypoints + Spatial pyramid (240-D vectors)
🤖 Unsupervised Clustering — PCA dimensionality reduction + K-Means grouping
🔎 Similarity Search — Find the most visually similar images to any query image
📊 Visualizations — Cluster grids, PCA scatter plots, elbow plots, contact sheets


🗂️ Project Structure
VisionSortAI/
│
├── VisionSort/                  # Input images folder
│
├── main.py                      # Main pipeline runner
├── preprocessing.py             # Noise removal + contrast enhancement
├── feature_extraction.py        # Feature vector computation
├── clustering.py                # PCA + K-Means clustering
├── similarity_search.py         # Nearest-neighbour similarity search
├── find_best_k.py               # Elbow method to find optimal clusters
├── requirements.txt             # Python dependencies
│
└── output/                      # Auto-generated results
    ├── preprocessed/            # Cleaned images
    ├── clusters/                # Images sorted into cluster folders
    ├── features.npz             # Cached feature matrix
    ├── cluster_assignments.csv  # Image → cluster label mapping
    ├── cluster_grid.png         # Thumbnail grid of all clusters
    ├── pca_scatter.png          # 2D PCA scatter plot
    ├── elbow_plot.png           # Optimal k analysis
    └── similarity_*.png         # Similarity search contact sheets

⚙️ How It Works
Raw Images
    │
    ▼
[1] Preprocessing
    Gaussian Blur (noise removal)
    CLAHE on LAB L-channel (contrast enhancement)
    │
    ▼
[2] Feature Extraction
    Colour Histogram  →  192-D
    ORB Mean Descriptor →  32-D
    Spatial Pyramid 4×4 →  16-D
    ─────────────────────────────
    Total Feature Vector: 240-D
    │
    ▼
[3] Clustering
    StandardScaler → PCA (50 components, ~98% variance)
    K-Means++ (k=3, configurable)
    Silhouette Score for quality evaluation
    │
    ▼
[4] Similarity Search
    BallTree nearest-neighbour on L2-normalised vectors
    Returns Top-N most visually similar images

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/Prachii345/VisionSort-AI.git
cd VisionSort-AI
2. Create a virtual environment
bashpython -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
3. Install dependencies
bashpip install -r requirements.txt
4. Add your images
Place all your images inside the VisionSort/ folder.
5. Run the pipeline
bashpython main.py
6. (Optional) Find the best number of clusters
bashpython find_best_k.py
Then open output/elbow_plot.png and update N_CLUSTERS in main.py.

🎛️ Configuration
Open main.py and adjust these settings at the top:
pythonINPUT_FOLDER = "VisionSort"    # Folder with your images
OUTPUT_FOLDER = "output"       # Where results are saved
N_CLUSTERS = 3                 # Number of clusters
N_SIMILAR = 5                  # Top-N similar images to retrieve
QUERY_IMAGE = None             # Set to an image path for similarity search

📊 Sample Results
MetricValueImages processed70Feature vector size240-DPCA variance retained98.7%Pipeline runtime~28 seconds

🛠️ Tech Stack
LibraryPurposeOpenCVImage preprocessing, ORB featuresNumPyNumerical computationsscikit-learnPCA, K-Means, BallTree, SilhouetteMatplotlibVisualizationstqdmProgress bars

📁 Output Examples

cluster_grid.png — Thumbnail grid showing all images grouped by visual similarity
pca_scatter.png — 2D scatter plot of all images coloured by cluster
elbow_plot.png — Inertia + silhouette score to pick optimal k
similarity_*.png — Contact sheet showing query image + top-5 matches


📜 License
This project is licensed under the MIT License.

👩‍💻 Author
Prachii345 — GitHub Profile

Built with ❤️ using Python and Computer Vision
