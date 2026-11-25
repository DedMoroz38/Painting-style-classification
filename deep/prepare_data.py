"""Build outputs/deep/index.csv: one row per image with its labels and splits.

Columns
    rel_path   path relative to data/complete_data
    cls        class (folder) name
    label13    0..12  (alphabetical class order)
    split13    train / val / test     (stratified 80/20, then 12.5 % of train -> val)
    label5     0..4 for the 5-class task, -1 if the image is not in it
    split5     train / val / test     for the 5-class task ('' if not in it)

The 5-class task uses exactly the same 5000 images and the same test images as
the traditional pipeline (src/dataset.py: split_indices on outputs/features.npz),
so results are directly comparable.

    /opt/anaconda3/bin/python deep/prepare_data.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CLASSES_13, CLASSES_5, DATA_DIR, INDEX_FILE, PROJECT_DIR, SEED

TRADITIONAL_FEATURES = PROJECT_DIR / "outputs" / "features.npz"
VAL_FRACTION_OF_TRAIN = 0.125   # 10 % of all images


def train_val_test(labels, test_idx_given=None):
    """Return an array of 'train'/'val'/'test' strings."""
    idx = np.arange(len(labels))
    if test_idx_given is None:
        train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    else:
        test_idx = test_idx_given
        train_idx = np.setdiff1d(idx, test_idx)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_FRACTION_OF_TRAIN,
                                          stratify=labels[train_idx], random_state=SEED)
    split = np.empty(len(labels), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    return split


def main():
    # ---- all images (13 classes)
    rows = []
    for label, cls in enumerate(CLASSES_13):
        for f in sorted(os.listdir(DATA_DIR / cls)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"rel_path": f"{cls}/{f}", "cls": cls, "label13": label})
    index = pd.DataFrame(rows)
    index["split13"] = train_val_test(index["label13"].to_numpy())

    # ---- 5-class task: replicate the traditional pipeline's split exactly
    trad = np.load(TRADITIONAL_FEATURES)
    trad_paths = [f"{Path(p).parent.name}/{Path(p).name}" for p in trad["paths"]]
    trad_labels = trad["labels"]
    _, trad_test_idx = train_test_split(np.arange(len(trad_labels)), test_size=0.2,
                                        stratify=trad_labels, random_state=SEED)
    trad_split = train_val_test(trad_labels, test_idx_given=trad_test_idx)

    lookup = {p: (lab, s) for p, lab, s in zip(trad_paths, trad_labels, trad_split)}
    index["label5"] = [lookup[p][0] if p in lookup else -1 for p in index["rel_path"]]
    index["split5"] = [lookup[p][1] if p in lookup else "" for p in index["rel_path"]]
    assert (index["label5"] >= 0).sum() == len(trad_paths), "some traditional images are missing"
    for p, lab in zip(trad_paths, trad_labels):   # sanity: same class names
        assert p.split("/")[0] == CLASSES_5[lab]

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(INDEX_FILE, index=False)

    print(f"{len(index)} images, {len(CLASSES_13)} classes -> {INDEX_FILE}")
    print(pd.crosstab(index["cls"], index["split13"]))
    five = index[index["label5"] >= 0]
    print("\n5-class task:")
    print(pd.crosstab(five["cls"], five["split5"]))


if __name__ == "__main__":
    main()
