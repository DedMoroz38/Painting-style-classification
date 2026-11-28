## Basically all preprocessing
import cv2
import numpy as np

from config import BORDER_CROP, MAX_SIDE

SIFT = cv2.SIFT_create()   # used only as a DoG keypoint detector (CV15), not for SIFT descriptors


# Preprocessing
def load_and_preprocess(path):
    img = cv2.imread(str(path))   # 3-channel BGR
    h, w = img.shape[:2]
    aspect_ratio = w / h

    # simple border crop
    dy, dx = int(h * BORDER_CROP), int(w * BORDER_CROP)
    img = img[dy:h - dy, dx:w - dx]


    # Resize so the longer side is MAX_SIDE
    # and then scale
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)

    return img, aspect_ratio


def grid_cells(image, rows=3, cols=3):
    # Split image into rows x cols blocks 
    h, w = image.shape[:2]
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)
    return [image[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] for r in range(rows) for c in range(cols)]


def mean_hue(hsv):
#    returns as (sin, cos) coming from how hue is in degrees/2
    angle = np.deg2rad(hsv[:, :, 0].astype(np.float32) * 2.0)
    s = hsv[:, :, 1].astype(np.float32) / 255
    weight = s.sum() + 1e-6   
    return (s * np.sin(angle)).sum() / weight, (s * np.cos(angle)).sum() / weight


def colour_histogram(hsv, bins=(8, 4, 4)):
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).flatten()
    return hist / hist.sum()


def global_statistics(bgr, hsv, aspect_ratio):
    s = hsv[:, :, 1].astype(np.float32) / 255
    v = hsv[:, :, 2].astype(np.float32) / 255

    hue_sin, hue_cos = mean_hue(hsv)
    hue_concentration = np.hypot(hue_sin, hue_cos)   # 1 = one dominant hue, 0 = all hues

    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colourfulness = (np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 255

    v_p5, v_p50, v_p95 = np.percentile(v, [5, 50, 95])

    return np.array([
        hue_sin, hue_cos, hue_concentration,
        s.mean(), s.std(),
        v.mean(), v.std(),
        v_p5, v_p50, v_p95,
        (v < 0.2).mean(),          # fraction of very dark pixels  (Baroque)
        (v > 0.8).mean(),          # fraction of very bright pixels (Rococo)
        colourfulness,
        np.log(aspect_ratio),      # log so that 2:1 and 1:2 are symmetric
        (s > 0.5).mean(),          # fraction of strongly saturated pixels
    ])


def colour_grid(hsv, rows=3, cols=3):
    features = []
    for cell in grid_cells(hsv, rows, cols):
        s = cell[:, :, 1].astype(np.float32) / 255
        v = cell[:, :, 2].astype(np.float32) / 255
        features += [*mean_hue(cell), s.mean(), s.std(), v.mean(), v.std()]
    return np.array(features)


def canny_edges(grey):
    # gaussian blur then median threshold
    #   return canny edge
    blurred = cv2.GaussianBlur(grey, (5, 5), 1.0)
    median = np.median(blurred)
    low, high = int(max(0, 0.66 * median)), int(min(255, 1.33 * median))
    return cv2.Canny(blurred, low, high)


def edge_density(edges):
    on = edges > 0
    return np.array([on.mean()] + [cell.mean() for cell in grid_cells(on)])


def gradient_orientation_histogram(grey_f, bins=8):
    """Sobel gradient directions (L6), weighted by gradient magnitude.
    Orientations are taken modulo 180 deg (an edge has no 'direction')."""
    gx = cv2.Sobel(grey_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(angle % 180, bins=bins, range=(0, 180), weights=magnitude)
    return hist / (hist.sum() + 1e-6)


def multiscale_gradient_stats(grey_f, sigmas=(1, 2, 4)):
    """Gradient magnitude mean/std after Gaussian smoothing at 3 scales (CV15
    scale space). Fine brushwork disappears at large sigma; big shapes do not."""
    features = []
    for sigma in sigmas:
        smooth = cv2.GaussianBlur(grey_f, (0, 0), sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        features += [magnitude.mean(), magnitude.std()]
    return np.array(features)


def fourier_texture(grey_f, size=256):
    """Fourier power spectrum (L3) split into 6 frequency bands + 4 orientation
    sectors. Each value is the fraction of the total power. High-frequency
    energy = visible brushwork / fine detail.

    A centred square crop is used so frequencies mean the same thing in every image.
    """
    h, w = grey_f.shape
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    patch = cv2.resize(grey_f[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)

    patch = patch - patch.mean()
    patch = patch * np.outer(np.hanning(size), np.hanning(size))   # window: avoid fake edges at the border
    power = np.abs(np.fft.fft2(patch)) ** 2

    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    radius = np.hypot(fx, fy)                      # cycles per pixel, 0 .. ~0.7
    angle = np.rad2deg(np.arctan2(fy, fx)) % 180   # orientation of the frequency

    valid = (radius > 0.01) & (radius <= 0.5)      # ignore DC / lowest frequencies and corners
    total = power[valid].sum() + 1e-12

    band_edges = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.5]
    bands = [power[valid & (radius > lo) & (radius <= hi)].sum() / total
             for lo, hi in zip(band_edges[:-1], band_edges[1:])]

    # Sectors centred on 0, 45, 90, 135 degrees
    shifted = (angle + 22.5) % 180
    sectors = [power[valid & (shifted >= a) & (shifted < a + 45)].sum() / total
               for a in (0, 45, 90, 135)]
    return np.array(bands + sectors)


# =====================================================================
# Group C: line structure  (L8-L9 Hough transform)
# =====================================================================
def hough_line_features(edges):
    """Straight lines found by the probabilistic Hough transform on the Canny map.
    Returns: lines per 10k pixels, straight-line length / edge pixels,
    and the length share of horizontal, vertical and diagonal lines."""
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=int(0.05 * max(h, w)), maxLineGap=5)
    if lines is None:   # about 3 % of the paintings have no straight lines
        return np.zeros(5)

    x1, y1, x2, y2 = lines[:, 0, :].T.astype(np.float32)
    lengths = np.hypot(x2 - x1, y2 - y1)
    angles = np.rad2deg(np.arctan2(y2 - y1, x2 - x1)) % 180

    horizontal = (angles < 15) | (angles > 165)
    vertical = (angles > 75) & (angles < 105)
    diagonal = ~(horizontal | vertical)
    total_length = lengths.sum()

    return np.array([
        len(lines) / (h * w) * 1e4,
        min(total_length / ((edges > 0).sum() + 1), 5.0),   # capped: overlapping segments can exceed 1
        lengths[horizontal].sum() / total_length,
        lengths[vertical].sum() / total_length,
        lengths[diagonal].sum() / total_length,
    ])


# =====================================================================
# Group D: interest points  (CV15 Harris & Stephens, Difference-of-Gaussian)
# =====================================================================
def harris_features(grey_f):
    """Harris corners: threshold the response, keep 3x3 local maxima (CV15).
    Returns corner density, mean and std of log corner strength."""
    response = cv2.cornerHarris(grey_f, blockSize=2, ksize=3, k=0.04)
    local_max = response == cv2.dilate(response, np.ones((3, 3), np.uint8))
    corners = (response > 0.01 * response.max()) & local_max
    strength = np.log10(response[corners] + 1e-12)
    h, w = grey_f.shape
    return np.array([corners.sum() / (h * w) * 1e4, strength.mean(), strength.std()])


def dog_blob_features(grey_u8):
    """DoG blob detector (CV15): extrema in a Difference-of-Gaussian scale space.
    Returns blob counts per 10k pixels in 5 size ranges = brushstroke-size distribution."""
    sizes = [kp.size for kp in SIFT.detect(grey_u8, None)]
    counts, _ = np.histogram(sizes, bins=[0, 4, 8, 16, 32, np.inf])
    h, w = grey_u8.shape
    return counts / (h * w) * 1e4


# =====================================================================
# Group E: regions  (CV13 k-means segmentation + connected components, CV14 shape)
# =====================================================================
def region_features(bgr, k=8, work_side=128, top_n=10):
    """Segment the image by k-means colour clustering in Lab (CV13), split each
    cluster into connected components (8-connectivity, CV13), then describe them
    with CV14 shape measures.

    Returns 6 numbers: number of regions, largest-region fraction, median region
    fraction, mean compactness and irregularity of the largest regions, and the
    share of the most common colour cluster.
    """
    # Work on a small copy: k-means on 260k pixels per image would be slow
    h, w = bgr.shape[:2]
    scale = work_side / max(h, w)
    small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    cv2.setRNGSeed(0)   # same clusters every run
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(lab, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])
    n_pixels = labels.size
    min_area = max(4, int(0.001 * n_pixels))   # ignore regions smaller than 0.1 % of the image

    # Connected components of every colour cluster
    regions = []   # (area, mask of that component)
    for cluster in range(k):
        mask = (labels == cluster).astype(np.uint8)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):                         # label 0 = background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                regions.append((area, comp == i))

    areas = np.array([a for a, _ in regions], dtype=np.float32)

    # Shape measures for the largest regions only (CV14)
    compactness, irregularity = [], []
    for area, mask in sorted(regions, key=lambda r: r[0], reverse=True)[:top_n]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
        compactness.append(4 * np.pi * area / (perimeter ** 2 + 1e-6))       # 1 for a circle
        ys, xs = np.nonzero(mask)
        max_r2 = ((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).max()
        irregularity.append(np.pi * max_r2 / area)                           # 1 for a disc, larger = spread out

    cluster_share = np.bincount(labels.ravel(), minlength=k) / n_pixels

    return np.array([
        len(regions),
        areas.max() / n_pixels,
        np.median(areas) / n_pixels,
        np.mean(compactness),
        np.log(np.mean(irregularity)),
        cluster_share.max(),
    ])


# =====================================================================
# Run everything
# =====================================================================
def extract_features(path):
    bgr, aspect_ratio = load_and_preprocess(path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey_f = grey.astype(np.float32) / 255
    edges = canny_edges(grey)

    return {
        "colour_hist": colour_histogram(hsv),
        "global_stats": global_statistics(bgr, hsv, aspect_ratio),
        "colour_grid": colour_grid(hsv),
        "edges_texture": np.concatenate([
            edge_density(edges),
            gradient_orientation_histogram(grey_f),
            multiscale_gradient_stats(grey_f),
            fourier_texture(grey_f),
        ]),
        "lines": hough_line_features(edges),
        "interest_points": np.concatenate([harris_features(grey_f), dog_blob_features(grey)]),
        "regions": region_features(bgr),
    }
