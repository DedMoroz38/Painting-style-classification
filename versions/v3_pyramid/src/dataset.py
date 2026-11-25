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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import DEEP_INDEX_FILE, FEATURES_FILE, HISTOGRAM_GROUPS, RANDOM_SEED, TASK, TEST_SIZE


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
    """Fixed train/test split. Always returns the same indices.

    5 classes : stratified 80/20 split.
    13 classes: the test images of the deep pipeline (outputs/deep/index.csv), so the
                two pipelines are compared on exactly the same test set. The deep
                pipeline's 'val' images are used here as training data (we tune by CV).
    """
    labels = data["labels"]
    indices = np.arange(len(labels))
    if TASK == "5":
        return train_test_split(indices, test_size=TEST_SIZE, stratify=labels,
                                random_state=RANDOM_SEED)
    index = pd.read_csv(DEEP_INDEX_FILE)
    test_paths = set(index.loc[index["split13"] == "test", "rel_path"])
    rel_paths = [f"{Path(p).parent.name}/{Path(p).name}" for p in data["paths"]]
    is_test = np.array([p in test_paths for p in rel_paths])
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
    "svm": {
        "pca": ["passthrough", PCA(n_components=0.95, random_state=RANDOM_SEED)],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1],
    },
    "rf": {   # trees don't need scaling or PCA
        "clf__max_features": ["sqrt", 0.2],
        "clf__min_samples_leaf": [1, 3],
    },
}

# 13 classes: an RBF SVM costs ~O(n^2), so the full grid would take many hours.
# Search a small grid around the 5-class winner (C=10, gamma='scale', no PCA).
if TASK == "13":
    PARAM_GRIDS["svm"] = {
        "pca": ["passthrough"],
        "clf__C": [3, 10, 30],
        "clf__gamma": ["scale"],
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
