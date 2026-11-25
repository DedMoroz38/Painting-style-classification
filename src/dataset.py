"""Shared helpers used by train.py, evaluate.py and ablation.py:

- load the saved features
- build the feature matrix X from a chosen list of feature groups
- make the (fixed) train/test split
- build the sklearn pipeline: scaling -> group weighting -> optional PCA -> classifier
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import FEATURES_FILE, HISTOGRAM_GROUPS, RANDOM_SEED, SPLIT_FILE


# ---------------------------------------------------------------- data
def load_features():
    """Load outputs/features.npz into a plain dict."""
    data = np.load(FEATURES_FILE)
    return {key: data[key] for key in data.files}


def build_matrix(data, groups):
    """Concatenate the chosen feature groups into one matrix X.

    Histogram groups get a square root (Hellinger mapping): this makes Euclidean
    distance between histograms behave better. It has no fitted parameters, so
    it cannot leak information between train and test.

    Returns X and the list of group sizes (needed for GroupWeighter).
    """
    blocks, sizes = [], []
    for group in groups:
        block = data[group].astype(np.float64)
        if group in HISTOGRAM_GROUPS:
            block = np.sqrt(block)
        blocks.append(block)
        sizes.append(block.shape[1])
    return np.hstack(blocks), sizes


def split_indices(data):
    """Fixed train/test split (80/20). Always returns the same indices.

    The test images are the ones marked 'test' in SPLIT_FILE, so the traditional
    and deep pipelines are compared on exactly the same test set. Everything else
    is training data (hyper-parameters are tuned by cross-validation on it).
    """
    index = pd.read_csv(SPLIT_FILE)
    test_images = set(index.loc[index["split13"] == "test", "rel_path"])
    image_names = [f"{Path(p).parent.name}/{Path(p).name}" for p in data["paths"]]   # "Class/file.jpg"
    is_test = np.array([name in test_images for name in image_names])
    indices = np.arange(len(image_names))
    return indices[~is_test], indices[is_test]


# ---------------------------------------------------------------- pipeline
class GroupWeighter(BaseEstimator, TransformerMixin):
    """Multiply each feature group by 1/sqrt(group size).

    After standardisation every column has variance 1, so a 128-dim group would
    otherwise count 128 times more than a 1-dim one in distances and PCA.
    With this weighting each group contributes the same total variance.
    """

    def __init__(self, group_sizes):
        self.group_sizes = group_sizes

    def fit(self, X, y=None):
        self.weights_ = np.concatenate([np.full(n, 1 / np.sqrt(n)) for n in self.group_sizes])
        return self

    def transform(self, X):
        return X * self.weights_


CLASSIFIERS = {
    "knn": KNeighborsClassifier(),
    "svm": SVC(kernel="rbf", cache_size=1000),   # bigger kernel cache = faster on large data
    "rf": RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1),
}

# Hyper-parameters tried by GridSearchCV in train.py.
# 'pca' = 'passthrough' means "no PCA"; PCA(0.95) keeps 95 % of the variance.
PARAM_GRIDS = {
    "knn": {
        "pca": ["passthrough", PCA(n_components=0.95, random_state=RANDOM_SEED)],
        "clf__n_neighbors": [1, 3, 5, 9, 15, 25],
        "clf__weights": ["uniform", "distance"],
    },
    "svm": {   # small grid: an RBF SVM on ~34k images takes minutes per fit
        "pca": ["passthrough"],
        "clf__C": [3, 10, 30],
        "clf__gamma": ["scale"],
    },
    "rf": {   # trees don't need scaling or PCA
        "clf__max_features": ["sqrt", 0.2],
        "clf__min_samples_leaf": [1, 3],
    },
}


def make_pipeline(model_name, group_sizes):
    """Standardise -> weight groups -> (PCA) -> classifier.

    Everything is inside one Pipeline, so during cross-validation the scaler
    and PCA are fitted on the training folds only (no data leakage).
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("weight", GroupWeighter(group_sizes)),
        ("pca", "passthrough"),
        ("clf", clone(CLASSIFIERS[model_name])),
    ])
