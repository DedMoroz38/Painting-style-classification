# Traditional CV for WikiArt Art-Movement Classification

Classify paintings into art movements using **only traditional (non-deep-learning) computer vision**: hand-crafted features from COMP3204 + a classical classifier.

- **Dataset**: Kaggle `sivarazadi/wikiart-art-movementsstyles` (29 GB zip, 13 classes, each stored as `Class/Class/*.jpg`)
- **Phase 1 scope**: 5 visually distinct classes (below). Extend to all 13 once the pipeline works.

---

## 1. Phase 1 classes

Chosen to be as far apart as possible in the cues hand-crafted features can actually measure (palette, brightness, flatness, line structure, texture):

| Class | Defining visual cues | Features expected to separate it |
|---|---|---|
| **Japanese_Art** | Flat colour areas, strong black outlines, beige/cream paper, lots of empty space, low texture | Low edge density inside regions, long clean contours, few large colour segments, beige-dominant hue histogram |
| **Western_Medieval** | Gold backgrounds, flat icon-like figures, saturated reds/blues, frontal symmetric layout | Gold/yellow hue at high saturation, grid colour layout (gold in upper cells), few blobs, flat regions |
| **Baroque** | Very dark backgrounds, dramatic chiaroscuro, warm browns, a few bright highlights | Low mean V, high V contrast, brown-dominant hue, dark border grid cells |
| **Rococo** | Light pastel palette, soft blended edges, pinks/sky blues, airy outdoor scenes | High V, low–medium S, pink/light-blue hues, low strong-edge density |
| **Expressionism** | Non-naturalistic saturated colour, thick visible brushwork, distorted forms | High S, wide hue spread, high texture energy at mid/high frequencies, many DoG blobs, many Harris corners |

**Available images** (checked on Kaggle): Baroque 5312, Expressionism 2607, Rococo 2521, Japanese_Art 2235, Western_Medieval 1158. Sampling **1000 per class** gives a balanced 5,000-image set.

**Expected hardest pair: Baroque vs Rococo** (same subjects, same oil-on-canvas look). Keeping it is deliberate: it tests whether the brightness/saturation features work. If you want an easier start, swap Rococo for **Art_Nouveau**, but note it overlaps with Japanese_Art (Japonisme) and with Medieval (Klimt's gold).

Classes deliberately **left out** for now: Academic_Art / Neoclassicism / Realism / Romanticism (all smooth naturalistic oil, heavily mutually confusable), Renaissance (between Medieval and Baroque), Primitivism (overlaps Expressionism), Symbolism (defined by subject, not by appearance).

---

## 2. Getting the data without downloading 29 GB

Kaggle has **no "download one folder" option**. In order of preference:

### Option A (recommended): build the subset on Kaggle, download only that
1. Create a Kaggle Notebook → *Add Input* → this dataset. It is mounted read-only under `/kaggle/input/`, so nothing is downloaded.
2. Run the script below. It samples N images per class, downsizes them and writes a single zip to `/kaggle/working/`.
3. Save version → download `wikiart5.zip` from the notebook's *Output* tab (or `kaggle kernels output <user>/<notebook>`). About 5,000 images at ≤768 px ≈ 0.5 GB.

```python
import os, random, zipfile, collections
import cv2

CLASSES = ["Japanese_Art", "Western_Medieval", "Baroque", "Rococo", "Expressionism"]
N_PER_CLASS = 1000        # balanced subset; lower it if a class has fewer images
MAX_SIDE = 768            # keep enough resolution for brushwork texture
EXT = (".jpg", ".jpeg", ".png")
random.seed(0)

# Find images by walking the mount (robust to the nested Class/Class/ layout)
paths = collections.defaultdict(list)
for root, _, files in os.walk("/kaggle/input"):
    cls = next((p for p in root.split(os.sep) if p in CLASSES), None)
    if cls:
        paths[cls] += [os.path.join(root, f) for f in files if f.lower().endswith(EXT)]
for c in CLASSES:
    print(c, len(paths[c]))  # check class sizes before choosing N_PER_CLASS

with zipfile.ZipFile("/kaggle/working/wikiart5.zip", "w", zipfile.ZIP_STORED) as zf:
    for c in CLASSES:
        for p in random.sample(sorted(paths[c]), min(N_PER_CLASS, len(paths[c]))):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue  # skip corrupt files
            h, w = img.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            zf.writestr(f"{c}/{os.path.basename(p)}", buf.tobytes())
```

Alternatively, run the whole feature extraction (Section 4) in the Kaggle notebook and download only `features.npz` (a few MB).

### Option B: per-file download with the Kaggle CLI / kagglehub
Both tools can fetch **individual files** but not folders:
```bash
kaggle datasets files sivarazadi/wikiart-art-movementsstyles --csv --page-size 1000   # list (paged)
kaggle datasets download sivarazadi/wikiart-art-movementsstyles -f "Baroque/Baroque/<file>.jpg"
```
```python
kagglehub.dataset_download("sivarazadi/wikiart-art-movementsstyles", path="Baroque/Baroque/<file>.jpg")
```
This is one HTTP request per image. Fine for a few hundred, but slow and may be rate-limited for thousands. You also get full-resolution files.

### Option C: download everything
Needs ~29 GB for the zip plus the extracted size. Only worth it if you will use all 13 classes later anyway.

---

## 3. Pipeline overview

```
image → preprocess → 5 feature groups (~280 dims) → per-group normalisation
      → [optional PCA] → classifier (kNN baseline, RBF-SVM main) → evaluation
```

All features are fixed-length per image, so no bag-of-words/codebook step is needed in Phase 1.

---

## 4. Pipeline details

### 4.1 Preprocessing
- **Resize keeping aspect ratio** so the longer side = 512 px (`INTER_AREA`). Do **not** squash to a square: it distorts shapes and stroke orientations. Store the original aspect ratio as a feature (portrait icons vs landscape scenes).
- **Crop a 3–5 % border.** Removes frames, scan margins and museum borders that otherwise dominate edge and colour features.
- **Colour spaces** (CV10, CV13): HSV for histograms (hue/saturation/value are directly meaningful for style), Lab for k-means segmentation (Euclidean distance ≈ perceptual difference), grey for edges/texture.
- **No global smoothing, no histogram equalisation.** Smoothing destroys brushwork texture; equalisation destroys the brightness distribution that separates Baroque from Rococo. Gaussian smoothing is applied only *inside* Canny / Harris / DoG where those methods require it (L5, L7, CV15).
- **No rotation augmentation and no rotation invariance.** Paintings have a canonical "up", and orientation is informative (verticals in icons, horizontals in landscapes).

### 4.2 Feature groups

#### A. Colour (CV10, CV13, L4) — about 195 dims
| Feature | Dims | Notes |
|---|---|---|
| Global joint HSV histogram, 8 H × 4 S × 4 V bins | 128 | L1-normalised (invariant to image size, CV13) |
| Colour statistics | ~12 | Circular hue mean/spread (via sin/cos of hue), mean/std of S and V, V percentiles (5/50/95), fraction of very dark and very bright pixels, colourfulness |
| 3×3 grid of mean/std of H(sin, cos), S, V | 9 × 6 = 54 | Cheap spatial layout (gold top in icons, dark surround in Baroque). Uses the CV13 grid-based idea without 9 full histograms |

#### B. Edges & texture (L3, L5, L6, L7, CV15) — about 55 dims
| Feature | Dims | Notes |
|---|---|---|
| Canny edge density, global + 3×3 grid | 10 | Flat (Japanese, Medieval) vs busy (Expressionism) |
| Sobel gradient-orientation histogram, magnitude-weighted | 8 | Global; 2×2 grid optional (+24) |
| Gradient magnitude mean/std at 3 Gaussian scales (σ = 1, 2, 4) | 6 | Scale-space idea from CV15: fine vs coarse brushwork |
| Fourier power spectrum: energy in 6 radial bands + 4 angular sectors, normalised | 10 | L3 explicitly lists texture recognition as a Fourier use; high-frequency energy = visible brushwork |

#### C. Line structure (L8, L9) — about 6 dims
- Probabilistic Hough lines on the Canny map: number of lines, total line length / edge pixels (how "straight" the structure is), and the share of near-horizontal / near-vertical / diagonal lines.
- Separates architectural, symmetric Medieval and outlined Japanese work from organic Rococo and Expressionism. It will matter more when Cubism-like classes are added later.

#### D. Interest points (CV15) — about 8 dims
- **Harris** (k ≈ 0.04–0.06): corner count per 10k pixels after thresholding + 3×3 non-max suppression, mean and std of the response.
- **DoG blobs**: blob counts per octave (4 octaves), normalised by image area. This is a *brushstroke-size distribution*, not object detection. Can be taken from OpenCV's SIFT detector keypoints (detector only, use `kp.size`) or built from a Gaussian pyramid directly.

#### E. Regions (CV13, CV14) — about 8 dims
1. k-means colour clustering in Lab (K = 8, optionally with a weak x,y term as in CV13) → segment labels.
2. Connected-component labelling (8-connectivity) on each cluster → regions; drop tiny regions (< 0.1 % of the image).
3. Features: number of regions, fraction of the image covered by the largest region, median region area, mean compactness and mean irregularity of the 10 largest regions (CV14), palette dominance (share of the biggest k-means cluster).

Flat-colour styles produce few, large, compact regions; painterly styles produce many small irregular ones.

### 4.3 Combining features
1. **Histograms**: L1-normalise; optionally apply a square root (Hellinger mapping, independent reading), which helps Euclidean-distance classifiers.
2. **Standardise** every dimension (zero mean, unit variance), fitted on training data only.
3. **Group weighting**: multiply each group by `1/sqrt(group_dims)`. Without this, the 195 colour dims would outvote the 8 interest-point dims in any distance or PCA. Tune these weights by CV if time allows.
4. **PCA** (CV12): fit on the training fold, keep ~95 % variance, and plot the explained-variance curve for the report. With ~280 dims and ~4,000 training images PCA is optional for SVM, so report results both with and without it.

### 4.4 Classifiers
*(Classical ML. Not covered in the provided lecture notes, so cite as independent reading and confirm the spec allows it.)*
- **kNN** (k tuned, Euclidean on the weighted features): baseline, directly comparable to the CV13 notion of feature-vector distance.
- **RBF-SVM** (tune C, γ by grid search): main model.
- **Random Forest**: optional, mainly for **permutation feature importance** to show which feature groups matter (no PCA before RF).

---

## 5. Evaluation protocol
- **Balanced sampling**: equal images per class (Section 2). If you later use imbalanced data, use `class_weight="balanced"`.
- **Split**: 80 % train / 20 % held-out test, stratified. If filenames encode the artist (WikiArt files are usually `artist-name_title.jpg`), use an **artist-disjoint split** (`StratifiedGroupKFold` / `GroupShuffleSplit`). Otherwise the classifier partly learns individual artists and accuracy is inflated.
- **Model selection**: 5-fold stratified CV on the training set only. Put scaler + weighting + PCA + classifier in one sklearn `Pipeline` so nothing leaks from validation folds.
- **Metrics**: accuracy, macro-F1, per-class precision/recall, confusion matrix. Expect Baroque↔Rococo to be the main confusion.
- **Test set** touched once, at the end.

---

## 6. Build order (each step = one row in an ablation table)

| Step | Add | Purpose |
|---|---|---|
| 0 | Data subset + feature caching to `.npz` | Never recompute features when tuning classifiers |
| 1 | Global HSV histogram + kNN | Baseline |
| 2 | Colour statistics + grid colour layout | Brightness/contrast and composition |
| 3 | Edge & texture group (B) | Flat vs painterly |
| 4 | Hough lines (C) + interest points (D) | Structure and stroke scale |
| 5 | Region features (E) | Flat-colour segmentation |
| 6 | Group weighting, PCA, switch to RBF-SVM, tune | Final model |
| 7 | Ablation: remove each group in turn | Which cues matter per class |

Report CV macro-F1 after every step, and show example misclassified images for the discussion.

---

## 7. Pitfalls checklist
- [ ] Fit scaler and PCA inside CV folds only (no leakage)
- [ ] Hue is circular: use sin/cos for statistics; red sits at both ends of the hue axis
- [ ] Some images are greyscale or have alpha/CMYK: force 3-channel BGR on load and skip unreadable files
- [ ] Borders/frames cropped before feature extraction
- [ ] Same artist not in both train and test (if artist is recoverable)
- [ ] Fixed random seeds for sampling, k-means and CV
- [ ] Normalise counts (corners, blobs, regions, lines) by image area

---

## 8. Lecture mapping

| Technique | Source |
|---|---|
| Colour spaces (RGB vs HSV) | CV10 |
| Point operators, histograms | Lecture 4 |
| Fourier transform, frequency content / texture | Lectures 2–3 |
| Gaussian and other group operators (convolution) | Lecture 5 |
| Sobel, Canny | Lectures 6–7 |
| Hough transform | Lectures 8–9 |
| PCA, covariance, eigendecomposition | CV12 |
| Global / grid / region / local features, joint colour histograms, Otsu, k-means segmentation, connected components | CV13 |
| Area, perimeter, compactness, irregularity, moments | CV14 |
| Harris corners, Gaussian scale space, DoG blobs | CV15 |
| kNN / SVM / Random Forest, Hellinger mapping, artist-disjoint evaluation | Independent reading |

Not used: mathematical morphology (erosion/dilation is not in these notes), rotation augmentation, and any learned/CNN features.

---

## 9. After Phase 1
1. Add the remaining 8 classes and expect a large accuracy drop driven by the naturalistic-oil cluster (Academic, Neoclassicism, Realism, Romanticism).
2. If local-feature lectures later cover descriptors and codebooks, add **bag-of-visual-words** (k-means codebook over local descriptors at Harris/DoG points) as a sixth feature group.
3. Re-run the ablation to see which groups still carry weight with 13 classes.
