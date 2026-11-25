"""Optional: figures and tables for the report (report/figures/).

Run after train.py and evaluate.py:
    python src/report_figures.py

1. class_counts.pdf      images per class                       (data visualisation)
2. examples.pdf          one painting per class                 (data visualisation)
3. confusion.pdf         row-normalised test confusion matrix of the final model
4. group_importance.pdf  per-class drop in F1 when one feature group is shuffled
                         (which feature group helps which class)
Also writes class_feature_means.csv: the average of interpretable features per class.
"""
import os

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

from config import CLASSES, MODELS_DIR, PROJECT_DIR, RANDOM_SEED, RESULTS_DIR
from dataset import build_matrix, load_features, split_indices

MODEL = "svm_no_grid"   # the final traditional model
FIG_DIR = PROJECT_DIR / "report" / "figures"
SHORT = [c.replace("Western_", "").replace("_Art", "").replace("_", " ") for c in CLASSES]
BLUES = LinearSegmentedColormap.from_list(   # one-hue sequential ramp, light -> dark
    "blues", ["#ffffff", "#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"])
TEXT = "#222222"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": "#888888",
                     "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT})


def save(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def class_counts(labels):
    counts = np.bincount(labels, minlength=len(CLASSES))
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.barh([SHORT[i] for i in order], counts[order], color="#2370cd", height=0.7)
    for y, i in enumerate(order):
        ax.text(counts[i] + 60, y, f"{counts[i]:,}", va="center", fontsize=6)
    ax.set_xlabel("images")
    ax.set_xlim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "class_counts.pdf")


def examples(paths, labels):
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(2, 7, figsize=(6.9, 2.25))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), range(len(CLASSES))):
        path = paths[rng.choice(np.flatnonzero(labels == c))]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        side = min(h, w)                                   # centre square crop for a tidy grid
        img = img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
        ax.imshow(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA))
        ax.set_title(SHORT[c], fontsize=7, pad=2)
    save(fig, "examples.pdf")


def confusion(y_true, y_pred):
    cm = 100 * confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if round(cm[i, j]) >= 1:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] > 50 else TEXT)
    ax.set_xticks(range(len(CLASSES)), SHORT, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(CLASSES)), SHORT, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    save(fig, "confusion.pdf")


def _shuffled_f1(model, X, y, cols, seed):
    """Per-class F1 after shuffling the columns `cols` across the test images."""
    X = X.copy()
    X[:, cols] = X[np.random.default_rng(seed).permutation(len(X))][:, cols]
    return f1_score(y, model.predict(X), labels=range(len(CLASSES)), average=None)


GROUP_LABELS = {"colour_hist": "A1 colour histogram", "global_stats": "A2 global colour stats",
                "colour_grid": "A3 colour grid", "edges_texture": "B edges & texture", "lines": "C lines",
                "interest_points": "D interest points", "regions": "E regions", "bovw": "F visual words"}


def group_importance(model, X, y, groups, sizes, base_f1, repeats=2):
    """Slow (16 SVM predictions), so the result is saved and reused on the next run."""
    csv = RESULTS_DIR / MODEL / "class_group_importance.csv"
    if csv.exists():
        table = pd.read_csv(csv, index_col=0)
    else:
        starts = np.cumsum([0] + sizes)
        jobs = [(g, slice(starts[k], starts[k + 1]), r) for k, g in enumerate(groups) for r in range(repeats)]
        f1s = Parallel(n_jobs=4)(delayed(_shuffled_f1)(model, X, y, cols, RANDOM_SEED + r) for _, cols, r in jobs)
        drops = pd.DataFrame([base_f1 - f for f in f1s], columns=CLASSES)
        drops["group"] = [g for g, _, _ in jobs]
        table = drops.groupby("group", sort=False).mean()
        table.to_csv(csv)

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.imshow(100 * table.clip(lower=0).to_numpy(), cmap=BLUES, vmin=0, aspect="auto")
    for i in range(len(table)):
        for j in range(len(CLASSES)):
            v = 100 * table.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 0.6 * 100 * table.to_numpy().max() else TEXT)
    ax.set_xticks(range(len(CLASSES)), SHORT, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(table)), [GROUP_LABELS[g] for g in table.index], fontsize=6)
    save(fig, "group_importance.pdf")
    return table


def feature_means(data, labels):
    """Average of a few interpretable features per class (used in the report's text)."""
    g, e, l, ip, r, cg = (data[k] for k in ["global_stats", "edges_texture", "lines",
                                             "interest_points", "regions", "colour_grid"])
    v_cells = cg[:, 4::6]   # mean brightness of each of the 9 grid cells
    f = pd.DataFrame({
        "dark_%": 100 * g[:, 10], "bright_%": 100 * g[:, 11], "saturated_%": 100 * g[:, 14],
        "colourfulness": g[:, 12], "aspect_w_h": np.exp(g[:, 13]),
        "edge_%": 100 * e[:, 0], "high_freq_%": 100 * (e[:, 28] + e[:, 29]),
        "lines_per_10k": l[:, 0], "vertical_line_%": 100 * l[:, 3],
        "harris_per_10k": ip[:, 0], "small_dog_per_10k": ip[:, 3] + ip[:, 4],
        "n_regions": r[:, 0], "largest_region": r[:, 1],
        "top_minus_bottom_V": v_cells[:, :3].mean(1) - v_cells[:, 6:].mean(1),
        "centre_minus_corner_V": v_cells[:, 4] - v_cells[:, [0, 2, 6, 8]].mean(1),
    })
    f["class"] = [CLASSES[i] for i in labels]
    means = f.groupby("class").mean()
    means.loc["ALL"] = f.drop(columns="class").mean()
    means.round(3).to_csv(RESULTS_DIR / MODEL / "class_feature_means.csv")
    return means


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_features()
    labels = data["labels"]
    bundle = joblib.load(MODELS_DIR / f"{MODEL}.joblib")
    X, sizes = build_matrix(data, bundle["groups"])
    _, test_idx = split_indices(data)
    X_test, y_test = X[test_idx], labels[test_idx]

    class_counts(labels)
    examples(data["paths"], labels)
    feature_means(data, labels)

    y_pred = bundle["model"].predict(X_test)
    confusion(y_test, y_pred)
    base_f1 = f1_score(y_test, y_pred, labels=range(len(CLASSES)), average=None)
    print(f"test macro-F1 {base_f1.mean():.3f}")
    print(group_importance(bundle["model"], X_test, y_test, bundle["groups"], sizes, base_f1).round(3))


if __name__ == "__main__":
    main()
