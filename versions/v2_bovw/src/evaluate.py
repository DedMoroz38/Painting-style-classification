"""Step 3: evaluate a trained model on the held-out test split (used only here).

    python src/evaluate.py --model svm

Writes to outputs/results/<model>/:
    report.txt               accuracy, macro-F1, per-class precision/recall/F1
    confusion_matrix.png     counts and row-normalised version
    misclassified.png        examples of wrong predictions (true -> predicted)
    group_importance.csv     drop in macro-F1 when each feature group is shuffled
"""
import argparse

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")   # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from config import CLASSES, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from dataset import build_matrix, load_features

SHORT_NAMES = [c.replace("Western_", "").replace("_Art", "") for c in CLASSES]  # for plot labels


def save_report(y_true, y_pred, bundle, out_dir):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    text = (
        f"Model: {bundle['model_name']}\n"
        f"Feature groups: {', '.join(bundle['groups'])}\n"
        f"Best params: {bundle['best_params']}\n"
        f"CV macro-F1 (training set): {bundle['cv_macro_f1']:.3f}\n\n"
        f"TEST accuracy: {acc:.3f}\n"
        f"TEST macro-F1: {macro_f1:.3f}\n\n"
        + classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    )
    (out_dir / "report.txt").write_text(text)
    print(text)
    return macro_f1


def save_confusion_matrix(y_true, y_pred, out_dir):
    big = len(CLASSES) > 5
    fig, axes = plt.subplots(1, 2, figsize=(26, 11) if big else (13, 5.5))
    for ax, normalize, title in [(axes[0], None, "Counts"), (axes[1], "true", "Row-normalised (recall)")]:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        ConfusionMatrixDisplay(cm, display_labels=SHORT_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d",
            xticks_rotation=45 if big else "horizontal")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_misclassified(paths, y_true, y_pred, out_dir, n=20):
    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(wrong, size=min(n, len(wrong)), replace=False)

    cols = 5
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), chosen):
        img = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"{SHORT_NAMES[y_true[i]]} -> {SHORT_NAMES[y_pred[i]]}", fontsize=10)
    fig.suptitle("Misclassified test images (true -> predicted)")
    fig.tight_layout()
    fig.savefig(out_dir / "misclassified.png", dpi=110)
    plt.close(fig)


def group_importance(model, X_test, y_test, group_sizes, groups, baseline_f1, repeats=5):
    """Permutation importance per feature GROUP: shuffle all columns of one group
    across the test images and see how much macro-F1 drops. Big drop = important."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    start = 0
    for group, size in zip(groups, group_sizes):
        cols = slice(start, start + size)
        drops = []
        for _ in range(repeats):
            X_shuffled = X_test.copy()
            X_shuffled[:, cols] = X_test[rng.permutation(len(X_test))][:, cols]
            drops.append(baseline_f1 - f1_score(y_test, model.predict(X_shuffled), average="macro"))
        rows.append({"group": group, "dims": size, "f1_drop_mean": np.mean(drops), "f1_drop_std": np.std(drops)})
        start += size
    return pd.DataFrame(rows).sort_values("f1_drop_mean", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["knn", "svm", "rf"], default="svm")
    args = parser.parse_args()

    bundle = joblib.load(MODELS_DIR / f"{args.model}.joblib")
    model = bundle["model"]

    # Test data: same groups and same split as used in training
    data = load_features()
    X, group_sizes = build_matrix(data, bundle["groups"])
    test_idx = bundle["test_idx"]
    X_test, y_test = X[test_idx], data["labels"][test_idx]
    paths_test = data["paths"][test_idx]

    y_pred = model.predict(X_test)

    out_dir = RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    test_f1 = save_report(y_test, y_pred, bundle, out_dir)
    save_confusion_matrix(y_test, y_pred, out_dir)
    save_misclassified(paths_test, y_test, y_pred, out_dir)

    repeats = 5 if len(y_test) < 2000 else 2   # SVM prediction on 8,500 images is slow
    importance = group_importance(model, X_test, y_test, group_sizes, bundle["groups"], test_f1, repeats)
    importance.to_csv(out_dir / "group_importance.csv", index=False)
    print("Feature-group importance (drop in test macro-F1 when the group is shuffled):")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
