"""All settings for the project in one place.

Change things here rather than inside the other scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "complete_data"     # one sub-folder per class
OUTPUT_DIR = PROJECT_DIR / "outputs"
FEATURES_FILE = OUTPUT_DIR / "features.npz"           # written by extract_features.py and bovw.py
MODELS_DIR = OUTPUT_DIR / "models"                    # written by train.py
RESULTS_DIR = OUTPUT_DIR / "results"                  # written by evaluate.py / ablation.py

# The train/test split comes from the deep pipeline's index file,
# so both pipelines are tested on exactly the same images.
SPLIT_FILE = OUTPUT_DIR / "deep" / "index.csv"

# ---------------------------------------------------------------- classes
CLASSES = ["Academic_Art", "Art_Nouveau", "Baroque", "Expressionism", "Japanese_Art",
           "Neoclassicism", "Primitivism", "Realism", "Renaissance", "Rococo",
           "Romanticism", "Symbolism", "Western_Medieval"]

# ---------------------------------------------------------------- preprocessing
BORDER_CROP = 0.04      # crop 4 % from every side (removes frames / scan margins)
MAX_SIDE = 512          # resize so the longer side is 512 px (aspect ratio kept)

# ---------------------------------------------------------------- feature groups
# Order matters: this is the order used when building the feature matrix.
# Each group is stored separately so we can switch groups on/off (ablation).
FEATURE_GROUPS = [
    "colour_hist",      # A  joint HSV histogram                        (CV13)
    "global_stats",     # A  colour statistics + aspect ratio           (CV10, L4)
    "colour_grid",      # A  3x3 grid of colour means / stds            (CV13)
    "edges_texture",    # B  Canny, Sobel, scale space, Fourier         (L3, L5-L7, CV15)
    "lines",            # C  Hough lines                                (L8-L9)
    "interest_points",  # D  Harris corners, DoG blobs                  (CV15)
    "regions",          # E  k-means segments + connected components    (CV13, CV14)
    "bovw",             # F  bag of visual words, whole image (bovw.py) (CV15, L5-L7, CV13)
    "bovw_grid",        # F  same, per 2x2 grid cell = spatial pyramid  (CV13 grid idea)
]

# Groups that are histograms: we take the square root of these (Hellinger mapping)
HISTOGRAM_GROUPS = ["colour_hist", "bovw", "bovw_grid"]

# ---------------------------------------------------------------- bag of visual words (bovw.py)
BOVW_WORDS = 500          # vocabulary size (k in k-means)
BOVW_MAX_POINTS = 600     # strongest DoG keypoints used per image
BOVW_VOCAB_IMAGES = 3000  # training images sampled to build the vocabulary

# ---------------------------------------------------------------- evaluation
RANDOM_SEED = 42
CV_FOLDS = 3            # cross-validation folds on the training set
