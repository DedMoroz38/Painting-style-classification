"""Step 1: compute features for every image and save them to outputs/features.npz.

Run once (takes a few minutes); training and evaluation then only read the .npz.

    python src/extract_features.py
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from config import CLASSES, DATA_DIR, FEATURES_FILE, FEATURE_GROUPS
from features import extract_features

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images():
    """Return (paths, labels) for every image, sorted so the order is always the same."""
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = DATA_DIR / cls
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
        paths += [folder / f for f in files]
        labels += [label] * len(files)
    return paths, np.array(labels)


def _init_worker():
    cv2.setNumThreads(1)   # each process uses one core; avoids oversubscription


def main():
    paths, labels = list_images()
    print(f"Found {len(paths)} images in {DATA_DIR}")

    start = time.time()
    results = []
    with ProcessPoolExecutor(initializer=_init_worker) as pool:
        for i, feats in enumerate(pool.map(extract_features, paths, chunksize=16), 1):
            results.append(feats)
            if i % 250 == 0:
                print(f"  {i}/{len(paths)} images  ({time.time() - start:.0f} s)")

    # Drop unreadable images
    keep = [i for i, f in enumerate(results) if f is not None]
    if len(keep) < len(paths):
        print(f"Skipped {len(paths) - len(keep)} unreadable images")

    # One matrix per feature group, plus labels and file paths
    groups = [g for g in FEATURE_GROUPS if g != "bovw"]   # "bovw" is added later by bovw.py
    arrays = {group: np.stack([results[i][group] for i in keep]) for group in groups}
    arrays["labels"] = labels[keep]
    arrays["paths"] = np.array([str(paths[i]) for i in keep])

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES_FILE, **arrays)

    print(f"\nSaved {FEATURES_FILE}  ({time.time() - start:.0f} s)")
    for group in groups:
        print(f"  {group:16s} {arrays[group].shape[1]:4d} dims")
    print(f"  {'TOTAL':16s} {sum(arrays[g].shape[1] for g in groups):4d} dims")


if __name__ == "__main__":
    main()
