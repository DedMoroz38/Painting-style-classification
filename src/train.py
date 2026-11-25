"""Step 2: train a classifier on the training split, tuning it with cross-validation.

The test split is NOT used here. evaluate.py uses it afterwards.

    python src/train.py --model knn     # baseline
    python src/train.py --model svm     # main model
    python src/train.py --model rf      # random forest

Optional: --groups colour_hist global_stats   (train on some feature groups only)
"""
import argparse
import time

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import CV_FOLDS, FEATURE_GROUPS, MODELS_DIR, RANDOM_SEED
from dataset import PARAM_GRIDS, build_matrix, load_features, make_pipeline, split_indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["knn", "svm", "rf"], default="svm")
    parser.add_argument("--groups", nargs="+", choices=FEATURE_GROUPS, default=FEATURE_GROUPS,
                        help="feature groups to use (default: all)")
    parser.add_argument("--name", help="save as <name>.joblib (default: the model name), "
                                       "e.g. --name svm_no_grid for a variant")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Data: features of the training images only
    data = load_features()
    X, group_sizes = build_matrix(data, args.groups)
    y = data["labels"]
    train_idx, test_idx = split_indices(data)
    X_train, y_train = X[train_idx], y[train_idx]
    print(f"Model: {args.model}   groups: {', '.join(args.groups)}")
    print(f"Training images: {len(train_idx)}   feature dims: {X.shape[1]}")

    # 2. Grid search with stratified k-fold cross-validation (CV_FOLDS), scored by macro-F1
    search = GridSearchCV(
        estimator=make_pipeline(args.model, group_sizes),
        param_grid=PARAM_GRIDS[args.model],
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
    )
    start = time.time()
    search.fit(X_train, y_train)   # refits the best setting on all training data at the end
    print(f"Grid search took {time.time() - start:.0f} s")

    # 3. Report the best settings
    results = pd.DataFrame(search.cv_results_)
    results["params"] = results["params"].astype(str)
    top = results.sort_values("rank_test_score").head(5)
    print("\nTop 5 settings (CV macro-F1, mean ± std over folds):")
    for _, row in top.iterrows():
        print(f"  {row.mean_test_score:.3f} ± {row.std_test_score:.3f}   {row.params}")
    print(f"\nBest CV macro-F1: {search.best_score_:.3f}")

    # 4. Save the trained model plus everything evaluate.py needs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or args.model
    out_file = MODELS_DIR / f"{name}.joblib"
    joblib.dump({
        "model": search.best_estimator_,
        "model_name": args.model,
        "groups": args.groups,
        "best_params": search.best_params_,
        "cv_macro_f1": search.best_score_,
        "test_idx": test_idx,
    }, out_file)
    results.to_csv(MODELS_DIR / f"{name}_cv_results.csv", index=False)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
