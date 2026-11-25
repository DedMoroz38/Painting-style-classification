"""Stage A, step 2: train classifiers on the frozen embeddings and find the hard classes.

1. Try several feature sets x heads (linear / MLP) x weight decay; pick the best by
   VALIDATION macro-F1 (test set untouched).
2. Hard-pair analysis on the validation set: which classes get confused most.
3. Specialist: a binary classifier for the worst pair, used only when the main
   model's top-2 classes are exactly that pair. Kept only if it helps on validation.
4. Predict the test set once with the chosen model -> predictions.csv (evaluate.py
   turns this into the report and plots).
5. Leakage check: test images that are near-duplicates of a training image.

    /opt/anaconda3/bin/python deep/probe.py --task 5
    /opt/anaconda3/bin/python deep/probe.py --task 13
"""
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from common import EMB_DIR, RESULTS_DIR, SEED, device, load_index, set_seed, task_table

FEATURE_SETS = {
    "clip": ["clip_global", "clip_tiles"],
    "dino": ["dino_global", "dino_tiles"],
    "clip+dino": ["clip_global", "clip_tiles", "dino_global", "dino_tiles"],
    "clip+dino+style": ["clip_global", "clip_tiles", "dino_global", "dino_tiles", "dino_style"],
}
HEADS = ["linear", "mlp"]
WEIGHT_DECAYS = [1e-4, 1e-2, 1e-1]
NEAR_DUPLICATE_COSINE = 0.97


# ---------------------------------------------------------------- data
def load_embeddings(rows, task):
    """Load the embedding file and return {name: array} aligned with `rows`."""
    path = EMB_DIR / "emb_all.npz"
    if not path.exists():
        path = EMB_DIR / f"emb_{task}.npz"
    emb = np.load(path)
    position = {p: i for i, p in enumerate(emb["rel_paths"])}
    missing = [p for p in rows["rel_path"] if p not in position]
    assert not missing, f"{len(missing)} images have no embedding, e.g. {missing[:3]}"
    order = np.array([position[p] for p in rows["rel_path"]])
    print(f"Embeddings: {path.name}")
    return {k: emb[k][order].astype(np.float32) for k in emb.files if k != "rel_paths"}


def build_features(emb, names):
    return np.hstack([emb[n] for n in names])


# ---------------------------------------------------------------- model
def make_head(kind, d_in, n_classes):
    if kind == "linear":
        return nn.Linear(d_in, n_classes)
    return nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(1024, n_classes))


def train_head(X_tr, y_tr, X_val, y_val, n_classes, kind, wd, max_epochs=100, patience=10):
    """Standardise -> head; class-weighted cross-entropy; early stopping on val macro-F1.
    Returns (predict_proba function, best val macro-F1)."""
    set_seed()
    dev = device()
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-6
    Xt = torch.tensor((X_tr - mean) / std, device=dev)
    Xv = torch.tensor((X_val - mean) / std, device=dev)
    yt = torch.tensor(y_tr, device=dev)

    counts = np.bincount(y_tr, minlength=n_classes)
    class_weight = torch.tensor(len(y_tr) / (n_classes * counts), dtype=torch.float32)
    head = make_head(kind, X_tr.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(dev), label_smoothing=0.1)

    best_f1, best_state, bad = -1, None, 0
    g = torch.Generator().manual_seed(SEED)
    for _ in range(max_epochs):
        head.train()
        for idx in torch.randperm(len(yt), generator=g).to(dev).split(256):
            opt.zero_grad()
            loss_fn(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            f1 = f1_score(y_val, head(Xv).argmax(1).cpu().numpy(), average="macro")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()

    def predict_proba(X):
        with torch.no_grad():
            return torch.softmax(head(torch.tensor((X - mean) / std, device=dev)), 1).cpu().numpy()
    return predict_proba, best_f1


# ---------------------------------------------------------------- analysis
def hard_pairs(y_true, y_pred, classes, top=8):
    """Most confused class pairs (both directions added), as fractions of each true class."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    pairs = [(classes[i], classes[j], cm[i, j] + cm[j, i], cm[i, j], cm[j, i])
             for i, j in itertools.combinations(range(len(classes)), 2)]
    df = pd.DataFrame(pairs, columns=["class_a", "class_b", "confusion", "a_as_b", "b_as_a"])
    return df.sort_values("confusion", ascending=False).head(top)


def apply_specialist(proba, a, b, specialist_proba):
    """Where the top-2 classes are {a, b}, let the specialist decide between them."""
    pred = proba.argmax(1)
    top2 = np.argsort(-proba, 1)[:, :2]
    mask = ((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a))
    pred = pred.copy()
    pred[mask] = np.where(specialist_proba[mask, 1] > 0.5, b, a)
    return pred, mask.sum()


def near_duplicates(emb, split):
    """Test images whose DINOv2 embedding is almost identical to a training image."""
    z = emb["dino_global"] / np.linalg.norm(emb["dino_global"], axis=1, keepdims=True)
    train, test = z[split == "train"], z[split == "test"]
    best = np.concatenate([(test[i:i + 512] @ train.T).max(1) for i in range(0, len(test), 512)])
    return best > NEAR_DUPLICATE_COSINE


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["5", "13"], required=True)
    args = parser.parse_args()

    rows, y, classes, split = task_table(load_index(), args.task)
    rows = rows.reset_index(drop=True)
    emb = load_embeddings(rows, args.task)
    tr, va, te = split == "train", split == "val", split == "test"
    n = len(classes)
    print(f"Task {args.task}: {tr.sum()} train / {va.sum()} val / {te.sum()} test, {n} classes\n")

    # 1. model selection on the validation set
    results, best = [], None
    for fs, head, wd in itertools.product(FEATURE_SETS, HEADS, WEIGHT_DECAYS):
        X = build_features(emb, FEATURE_SETS[fs])
        predict, val_f1 = train_head(X[tr], y[tr], X[va], y[va], n, head, wd)
        results.append({"features": fs, "head": head, "weight_decay": wd, "val_macro_f1": val_f1})
        print(f"  {fs:16s} {head:6s} wd={wd:<6g}  val macro-F1 = {val_f1:.3f}", flush=True)
        if best is None or val_f1 > best["f1"]:
            best = {"f1": val_f1, "features": fs, "head": head, "wd": wd, "predict": predict, "X": X}
    print(f"\nBest: {best['features']} / {best['head']} / wd={best['wd']}  (val macro-F1 {best['f1']:.3f})")

    out_dir = RESULTS_DIR / f"probe_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "model_selection.csv", index=False)

    # 2. hard pairs (validation set)
    X = best["X"]
    val_proba = best["predict"](X[va])
    pairs = hard_pairs(y[va], val_proba.argmax(1), classes)
    pairs.to_csv(out_dir / "hard_pairs_val.csv", index=False)
    print("\nMost confused pairs (validation):")
    print(pairs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3. specialist for the worst pair
    a, b = classes.index(pairs.iloc[0].class_a), classes.index(pairs.iloc[0].class_b)
    pair_tr, pair_va = tr & np.isin(y, [a, b]), va & np.isin(y, [a, b])
    spec_predict, _ = train_head(X[pair_tr], (y[pair_tr] == b).astype(int),
                                 X[pair_va], (y[pair_va] == b).astype(int), 2, best["head"], best["wd"])
    val_pred_spec, n_switched = apply_specialist(val_proba, a, b, spec_predict(X[va]))
    f1_plain = f1_score(y[va], val_proba.argmax(1), average="macro")
    f1_spec = f1_score(y[va], val_pred_spec, average="macro")
    use_specialist = f1_spec > f1_plain
    print(f"\nSpecialist {classes[a]} vs {classes[b]}: val macro-F1 {f1_plain:.3f} -> {f1_spec:.3f} "
          f"({n_switched} val images routed)  -> {'USED' if use_specialist else 'not used'}")

    # 4. test predictions (test set used once, here)
    test_proba = best["predict"](X[te])
    test_pred = test_proba.argmax(1)
    if use_specialist:
        test_pred, _ = apply_specialist(test_proba, a, b, spec_predict(X[te]))

    # 5. leakage check
    dup = near_duplicates(emb, split)
    print(f"\nNear-duplicates of a training image in the test set: {dup.sum()} / {te.sum()}")

    pred_df = pd.DataFrame({"rel_path": rows["rel_path"][te].values, "true": y[te], "pred": test_pred,
                            "near_duplicate": dup})
    for i, c in enumerate(classes):
        pred_df[f"p_{c}"] = test_proba[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "config.txt").write_text(
        f"features={best['features']}\nhead={best['head']}\nweight_decay={best['wd']}\n"
        f"val_macro_f1={best['f1']:.4f}\nspecialist={classes[a]} vs {classes[b]} used={use_specialist}\n")
    print(f"Saved {out_dir / 'predictions.csv'}  ->  now run: deep/evaluate.py --run probe_{args.task}")


if __name__ == "__main__":
    main()
