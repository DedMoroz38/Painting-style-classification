# COMPUTE FEATURES AND SAVE TO output/FEATURES_FILE (should be npz)

import os

import numpy as np

from config import CLASSES, DATA_DIR, FEATURES_FILE, FEATURE_GROUPS
from features import extract_features

# The bag-of-visual-words groups are added later by bovw.py
GROUPS = [g for g in FEATURE_GROUPS if not g.startswith("bovw")]


def main():
    # 1 Find all images
    paths, labels = [], []
    for label, class_name in enumerate(CLASSES):
        folder = DATA_DIR / class_name # since class name = folder name get folder name
        for file_name in sorted(os.listdir(folder)): 
            paths.append(str(folder / file_name))
            labels.append(label)
    print(f"Found {len(paths)} images")

    # 2 Compute faetures of each image
    features = {group: [] for group in GROUPS}
    for i, path in enumerate(paths):
        image_features = extract_features(path)
        for group in GROUPS:
            features[group].append(image_features[group])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)} images done")

    # 3. Save
    #one matrix per feature group 
    arrays = {group: np.array(features[group], dtype=np.float32) for group in GROUPS}
    arrays["labels"] = np.array(labels)
    arrays["paths"] = np.array(paths)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES_FILE, **arrays)

    print(f"Saved {FEATURES_FILE}")
    for group in GROUPS:
        print(f"  {group:16s} {arrays[group].shape[1]} numbers")


if __name__ == "__main__":
    main()
