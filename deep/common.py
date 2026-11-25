"""Shared settings and helpers for the deep-learning baseline.

Run every script in deep/ with the anaconda Python (it has PyTorch + MPS):
    /opt/anaconda3/bin/python deep/<script>.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "complete_data"        # all 42.5k images, one folder per class
OUT_DIR = PROJECT_DIR / "outputs" / "deep"
INDEX_FILE = OUT_DIR / "index.csv"                        # written by prepare_data.py
EMB_DIR = OUT_DIR / "embeddings"                          # written by embed.py
RESULTS_DIR = OUT_DIR / "results"

# 13 classes in alphabetical order (folder names)
CLASSES_13 = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []
# 5-class subset, same order as the traditional pipeline (src/config.py)
CLASSES_5 = ["Japanese_Art", "Western_Medieval", "Baroque", "Rococo", "Expressionism"]

SEED = 42


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_index():
    return pd.read_csv(INDEX_FILE)


def task_table(index, task):
    """Rows, labels, class names and split column for task '5' or '13'."""
    if task == "13":
        return index, index["label13"].to_numpy(), CLASSES_13, index["split13"].to_numpy()
    rows = index[index["label5"] >= 0]
    return rows, rows["label5"].to_numpy(), CLASSES_5, rows["split5"].to_numpy()


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
