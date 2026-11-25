"""Step 1b: Bag of Visual Words feature groups ("bovw", "bovw_grid"), added to the features file.

Built only from techniques in the lecture notes:
    1. points      : DoG keypoints (CV15), the strongest MAX_POINTS per image
    2. descriptor  : a hand-written gradient-orientation histogram of the patch around
                     each point - Sobel gradients (L5-L7), 4x4 cells x 8 orientations = 128
                     numbers (the same idea as the SIFT descriptor, but our own code)
    3. vocabulary  : k-means (CV13) on descriptors from TRAINING images only -> visual words
    4. encoding    : each image = normalised histogram of its visual words   -> "bovw"
    5. spatial pyramid: the same histogram for each cell of a 2x2 grid        -> "bovw_grid"
                     (WHERE the words occur: e.g. sky at the top, figures in the centre)

Run after extract_features.py (it needs the train/test split and the image paths):
    python src/bovw.py              # 5 classes
    TASK=13 python src/bovw.py      # 13 classes
"""
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import (BOVW_MAX_POINTS, BOVW_VOCAB_IMAGES, BOVW_WORDS, FEATURES_FILE, OUTPUT_DIR,
                    RANDOM_SEED, TASK)
from dataset import load_features, split_indices
from features import load_and_preprocess

CELLS, BINS = 4, 8          # 4x4 grid of cells, 8 orientation bins per cell -> 128 numbers
DESC_DIM = CELLS * CELLS * BINS


# ---------------------------------------------------------------- descriptors
NO_POINTS = (np.zeros((0, DESC_DIM), np.float32), np.zeros((0, 2), np.float32))


def describe_image(path):
    """Return (descriptors (n, 128), positions (n, 2) as (x, y) fractions of the image size)."""
    bgr, _ = load_and_preprocess(path)
    if bgr is None:
        return NO_POINTS
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. DoG keypoints (CV15); OpenCV's SIFT *detector* only, as in features.py
    keypoints = cv2.SIFT_create().detect(grey, None)
    keypoints = sorted(keypoints, key=lambda kp: -kp.response)[:BOVW_MAX_POINTS]

    # 2. Sobel gradients for the whole image, once (L5-L7)
    g = grey.astype(np.float32) / 255
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation_bin = ((np.arctan2(gy, gx) + np.pi) / (2 * np.pi) * BINS).astype(int) % BINS

    h, w = grey.shape
    descriptors, positions = [], []
    for kp in keypoints:
        x, y = kp.pt
        r = int(np.clip(kp.size * 1.5, 8, 32))          # patch half-size follows the blob size
        x0, y0 = int(x) - r, int(y) - r
        if x0 < 0 or y0 < 0 or x0 + 2 * r > w or y0 + 2 * r > h:
            continue                                     # skip patches that leave the image
        mag = magnitude[y0:y0 + 2 * r, x0:x0 + 2 * r]
        ori = orientation_bin[y0:y0 + 2 * r, x0:x0 + 2 * r]
        # which of the 4x4 cells each pixel belongs to
        cell_1d = np.arange(2 * r) * CELLS // (2 * r)
        cell = cell_1d[:, None] * CELLS + cell_1d[None, :]
        # magnitude-weighted orientation histogram per cell, all cells in one bincount
        d = np.bincount((cell * BINS + ori).ravel(), weights=mag.ravel(), minlength=DESC_DIM)
        # SIFT-style normalisation: unit length, clip large values (robust to strong edges), renormalise
        d = d / (np.linalg.norm(d) + 1e-7)
        d = np.minimum(d, 0.2)
        descriptors.append(d / (np.linalg.norm(d) + 1e-7))
        positions.append((x / w, y / h))
    if not descriptors:
        return NO_POINTS
    return np.array(descriptors, np.float32), np.array(positions, np.float32)


def sample_descriptors(path, per_image=100):
    """A random subset of one image's descriptors (for building the vocabulary)."""
    d, _ = describe_image(path)
    rng = np.random.default_rng(zlib.crc32(str(path).encode()))   # same sample on every run
    return d[rng.permutation(len(d))[:per_image]]


# ---------------------------------------------------------------- encoding
_VOCAB = None


def _init_worker(vocab):
    global _VOCAB
    cv2.setNumThreads(1)
    _VOCAB = vocab


def word_histogram(path):
    """Assign each descriptor to its nearest visual word.

    Returns (whole-image histogram (K,), 2x2 grid histograms (4K,)). Both are divided
    by the image's total number of points, so the 4 grid cells together sum to 1.
    """
    k = len(_VOCAB)
    d, pos = describe_image(path)
    if len(d) == 0:
        return np.zeros(k, np.float32), np.zeros(4 * k, np.float32)
    dist = (d ** 2).sum(1, keepdims=True) - 2 * d @ _VOCAB.T + (_VOCAB ** 2).sum(1)
    words = dist.argmin(1)
    cell = (pos[:, 1] >= 0.5).astype(int) * 2 + (pos[:, 0] >= 0.5).astype(int)   # 0..3
    whole = np.bincount(words, minlength=k) / len(words)
    grid = np.bincount(cell * k + words, minlength=4 * k) / len(words)
    return whole.astype(np.float32), grid.astype(np.float32)


# ---------------------------------------------------------------- main
def main():
    data = load_features()
    paths = data["paths"]
    train_idx, _ = split_indices(data)       # vocabulary from training images only (no leakage)
    start = time.time()

    # 1. Vocabulary: k-means on descriptors from a random sample of training images
    rng = np.random.default_rng(RANDOM_SEED)
    vocab_paths = paths[rng.choice(train_idx, size=min(BOVW_VOCAB_IMAGES, len(train_idx)), replace=False)]
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(None,)) as pool:
        sample = np.vstack(list(pool.map(sample_descriptors, vocab_paths, chunksize=16)))
    print(f"Vocabulary: k-means with k={BOVW_WORDS} on {len(sample)} descriptors "
          f"from {len(vocab_paths)} training images  ({time.time() - start:.0f} s)")
    kmeans = MiniBatchKMeans(n_clusters=BOVW_WORDS, batch_size=4096, n_init=3,
                             random_state=RANDOM_SEED).fit(sample)
    vocab = kmeans.cluster_centers_.astype(np.float32)
    np.save(OUTPUT_DIR / f"bovw_vocabulary_{TASK}.npy", vocab)

    # 2. Encode every image as histograms of visual words
    whole, grid = [], []
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(vocab,)) as pool:
        for i, (h, g) in enumerate(pool.map(word_histogram, paths, chunksize=16), 1):
            whole.append(h)
            grid.append(g)
            if i % 5000 == 0:
                print(f"  {i}/{len(paths)} images  ({time.time() - start:.0f} s)", flush=True)
    data["bovw"], data["bovw_grid"] = np.stack(whole), np.stack(grid)
    print(f"Images with no keypoints: {(data['bovw'].sum(1) == 0).sum()}")

    # 3. Add the new groups to the features file (same row order as the other groups)
    np.savez_compressed(FEATURES_FILE, **data)
    print(f"Added 'bovw' ({data['bovw'].shape[1]} dims) and 'bovw_grid' ({data['bovw_grid'].shape[1]} dims) "
          f"to {FEATURES_FILE}  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
