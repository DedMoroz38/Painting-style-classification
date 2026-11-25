"""All settings for the project in one place.

Change things here rather than inside the other scripts.

Two tasks, chosen with the TASK environment variable (default 5):
    python src/train.py --model svm             # 5 classes  (data/wikiart5)
    TASK=13 python src/train.py --model svm     # 13 classes (data/complete_data)
"""
import os
from pathlib import Path

TASK = os.environ.get("TASK", "5")
assert TASK in ("5", "13"), "TASK must be 5 or 13"

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
if TASK == "5":
    DATA_DIR = PROJECT_DIR / "data" / "wikiart5"      # one sub-folder per class
    FEATURES_FILE = OUTPUT_DIR / "features.npz"       # written by extract_features.py
    MODELS_DIR = OUTPUT_DIR / "models"                # written by train.py
    RESULTS_DIR = OUTPUT_DIR / "results"              # written by evaluate.py / ablation.py
else:
    DATA_DIR = PROJECT_DIR / "data" / "complete_data"
    FEATURES_FILE = OUTPUT_DIR / "features_13.npz"
    MODELS_DIR = OUTPUT_DIR / "models_13"
    RESULTS_DIR = OUTPUT_DIR / "results_13"

# For 13 classes the test set is taken from the deep pipeline's index, so both
# pipelines are tested on exactly the same 8,500 images.
DEEP_INDEX_FILE = OUTPUT_DIR / "deep" / "index.csv"

# ---------------------------------------------------------------- classes
if TASK == "5":
    CLASSES = ["Japanese_Art", "Western_Medieval", "Baroque", "Rococo", "Expressionism"]
else:
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
]

# Groups that are histograms: we take the square root of these (Hellinger mapping)
HISTOGRAM_GROUPS = ["colour_hist"]


# ---------------------------------------------------------------- evaluation
RANDOM_SEED = 42
TEST_SIZE = 0.2         # 80 % train / 20 % held-out test
CV_FOLDS = 5 if TASK == "5" else 3   # CV folds on the training set (3 for 13 classes: 7x more data)
