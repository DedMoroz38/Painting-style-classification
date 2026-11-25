"""Step 4: ablation study, i.e. which feature groups actually help?

Uses 5-fold CV on the TRAINING split only (the test split is never touched),
with one fixed classifier so that only the features change between rows.

Part 1, "build-up": add feature groups one step at a time (plan section 6).
Part 2, "leave-one-out": all groups, minus one group at a time.

    python src/ablation.py              # RBF-SVM (C=10, gamma='scale')
    python src/ablation.py --model knn  # kNN (k=9, distance-weighted)

Writes outputs/results/ablation_<model>.csv
"""
import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import CV_FOLDS, FEATURE_GROUPS, RANDOM_SEED, RESULTS_DIR
from dataset import build_matrix, load_features, make_pipeline, split_indices

# Fixed (not tuned) settings, so each row is a fair comparison
FIXED_PARAMS = {
    "svm": {"clf__C": 10, "clf__gamma": "scale"},
    "knn": {"clf__n_neighbors": 9, "clf__weights": "distance"},
}

# Build-up order from the plan: each step adds groups to the previous one
BUILD_UP = [
    ("1 colour histogram", ["colour_hist"]),
    ("2 + colour stats & grid", ["global_stats", "colour_grid"]),
    ("3 + edges & texture", ["edges_texture"]),
    ("4 + lines & interest points", ["lines", "interest_points"]),
    ("5 + regions", ["regions"]),
]


def cv_score(data, groups, train_idx, model_name):
    X, group_sizes = build_matrix(data, groups)
    y = data["labels"]
    pipeline = make_pipeline(model_name, group_sizes).set_params(**FIXED_PARAMS[model_name])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipeline, X[train_idx], y[train_idx], cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean(), scores.std(), X.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(FIXED_PARAMS), default="svm")
    args = parser.parse_args()

    data = load_features()
    train_idx, _ = split_indices(data)
    rows = []

    print(f"Build-up ({args.model}, CV macro-F1 on training split)")
    groups = []
    for name, new_groups in BUILD_UP:
        groups = groups + new_groups
        mean, std, dims = cv_score(data, groups, train_idx, args.model)
        rows.append({"experiment": "build-up", "setting": name, "dims": dims, "cv_f1": mean, "cv_f1_std": std})
        print(f"  {name:32s} dims={dims:4d}   F1 = {mean:.3f} ± {std:.3f}")

    print(f"\nLeave-one-group-out ({args.model})")
    for left_out in FEATURE_GROUPS:
        groups = [g for g in FEATURE_GROUPS if g != left_out]
        mean, std, dims = cv_score(data, groups, train_idx, args.model)
        rows.append({"experiment": "leave-one-out", "setting": f"without {left_out}",
                     "dims": dims, "cv_f1": mean, "cv_f1_std": std})
        print(f"  without {left_out:24s} dims={dims:4d}   F1 = {mean:.3f} ± {std:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"ablation_{args.model}.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")


if __name__ == "__main__":
    main()
