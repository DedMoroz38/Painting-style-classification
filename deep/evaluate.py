"""Test-set report for any deep run (reads <run>/predictions.csv).

    /opt/anaconda3/bin/python deep/evaluate.py --run probe_5
    /opt/anaconda3/bin/python deep/evaluate.py --run probe_13
    /opt/anaconda3/bin/python deep/evaluate.py --run finetune_13

Writes report.txt, confusion_matrix.png and misclassified.png next to predictions.csv.
Metrics are also given without near-duplicate test images (leakage check).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from common import CLASSES_13, CLASSES_5, DATA_DIR, PROJECT_DIR, RESULTS_DIR, SEED


def short(c):
    return c.replace("Western_", "").replace("_Art", "").replace("_", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder name in outputs/deep/results/")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run
    df = pd.read_csv(run_dir / "predictions.csv")
    classes = CLASSES_5 if args.run.endswith("_5") else CLASSES_13
    y, p = df["true"].to_numpy(), df["pred"].to_numpy()

    text = f"Run: {args.run}\n"
    if (run_dir / "config.txt").exists():
        text += (run_dir / "config.txt").read_text()
    text += (f"\nTEST accuracy: {accuracy_score(y, p):.3f}\n"
             f"TEST macro-F1: {f1_score(y, p, average='macro'):.3f}\n")
    if "near_duplicate" in df:
        clean = ~df["near_duplicate"].to_numpy()
        text += (f"Without {(~clean).sum()} near-duplicate test images: "
                 f"accuracy {accuracy_score(y[clean], p[clean]):.3f}, "
                 f"macro-F1 {f1_score(y[clean], p[clean], average='macro'):.3f}\n")
    trad = PROJECT_DIR / "outputs" / "results" / "svm_no_grid" / "report.txt"   # final traditional model
    if classes == CLASSES_13 and trad.exists():
        line = next(l for l in trad.read_text().splitlines() if l.startswith("TEST macro-F1"))
        text += f"(Traditional SVM, same test set: {line.split(':')[1].strip()} macro-F1)\n"
    text += "\n" + classification_report(y, p, labels=range(len(classes)), target_names=classes, digits=3)
    (run_dir / "report.txt").write_text(text)
    print(text)

    # Confusion matrix (row-normalised)
    size = 5.5 if len(classes) == 5 else 11
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    cm = confusion_matrix(y, p, labels=range(len(classes)), normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=[short(c) for c in classes]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".2f", xticks_rotation=45)
    ax.set_title(f"{args.run}: row-normalised confusion matrix")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Misclassified examples
    wrong = np.flatnonzero(y != p)
    if len(wrong):
        chosen = np.random.default_rng(SEED).choice(wrong, size=min(20, len(wrong)), replace=False)
        fig, axes = plt.subplots(4, 5, figsize=(15, 13))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), chosen):
            ax.imshow(Image.open(DATA_DIR / df["rel_path"][i]).convert("RGB"))
            ax.set_title(f"{short(classes[y[i]])} -> {short(classes[p[i]])}", fontsize=10)
        fig.suptitle("Misclassified test images (true -> predicted)")
        fig.tight_layout()
        fig.savefig(run_dir / "misclassified.png", dpi=100)
        plt.close(fig)
    print(f"Saved report and plots to {run_dir}")


if __name__ == "__main__":
    main()
