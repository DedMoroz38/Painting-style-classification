# WikiArt painting-style classification

This project compares two ways of classifying painting style:

- **Traditional pipeline (`src/`).** Hand-crafted computer-vision features with an SVM. No neural networks.
- **Deep baseline (`deep/`).** Frozen CLIP and DINOv2 embeddings with a small MLP head. It is the upper bound the traditional pipeline is measured against.

The data is the Kaggle dataset [`sivarazadi/wikiart-art-movementsstyles`](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles): 42,499 images in 13 styles. The 5-class task is a subset of 1,000 images per style: Japanese, Medieval, Baroque, Rococo and Expressionism.

## Results (test set, macro-F1)

| Model | 5 classes | 13 classes |
|---|---|---|
| Traditional SVM, v1: hand-crafted features | 0.731 | 0.571 |
| Traditional SVM, v2: + 200 visual words | 0.760 | 0.596 |
| **Traditional SVM, v3: 500 visual words (final)** | **0.769** | **0.610** |
| Deep: CLIP ViT-L/14 + MLP | 0.959 | 0.901 |

Both pipelines are scored on exactly the same test images. Each style is classified from what the image looks like alone: no artist, date or title is used. Splits are random, and artists are not kept separate between training and test.

## Traditional pipeline (`src/`)

| File | What it does |
|---|---|
| `config.py` | All settings: paths, classes, image size, feature groups, visual-word settings, CV folds. |
| `features.py` | The hand-crafted feature groups: colour, edges and texture, Hough lines, Harris and DoG points, k-means regions. |
| `extract_features.py` | Step 1. Writes `outputs/features*.npz`. |
| `bovw.py` | Step 1b. Bag of visual words built from DoG points, hand-written Sobel descriptors and a k-means vocabulary. |
| `dataset.py` | Loads features, defines the fixed train/test split, and builds the pipeline (scaling, group weights, PCA, classifier). |
| `train.py` | Step 2. Grid search with cross-validation on the training set only. |
| `evaluate.py` | Step 3. One-off test-set report, confusion matrix and feature-group importance. |
| `ablation.py` | Measures which feature groups help. |
| `report_figures.py` | Optional. Figures and tables for the report (`report/figures/`), including the per-class feature-group importance. |

Run from the project root:

```bash
python src/extract_features.py
python src/bovw.py
python src/train.py --model svm --name svm_no_grid \
    --groups colour_hist global_stats colour_grid edges_texture lines interest_points regions bovw
python src/evaluate.py --model svm_no_grid
```

The code covers the 13-class task. The v1, v2 and 5-class results in the table come from earlier code, which is kept in the first commit of this repository.

## Deep baseline (`deep/`)

| Step | Command |
|---|---|
| Build the image index and splits | `python deep/prepare_data.py` |
| Compute the embeddings (GPU; see `deep/kaggle_embed.md`) | `python deep/embed.py ...` |
| Train the classifier head and analyse hard pairs | `python deep/probe.py --task 13` |
| Write the test-set report | `python deep/evaluate.py --run probe_13` |

## Not in the repo

The following are left out because of their size or copyright. They can all be rebuilt by running the scripts.

- `data/`
- extracted features
- trained models
- embeddings
- images of misclassified paintings

