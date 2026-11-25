"""Stage A, step 1: frozen embeddings with CLIP ViT-L/14 and DINOv2 ViT-L/14.

Self-contained (no imports from this project) so the SAME file runs locally and on Kaggle.

For every image we build 5 views (all 224x224):
    global  : resize shorter side to 224, centre crop          (whole composition)
    4 tiles : resize shorter side to 448, centre crop, 2x2 grid (brushwork at 2x detail)

Output: one .npz with
    rel_paths                          "Class/file.jpg" (used to match rows to index.csv)
    clip_global, clip_tiles            CLIP image embedding (768): global / mean of 4 tiles
    dino_global, dino_tiles            DINOv2 CLS token (1024):    global / mean of 4 tiles
    dino_style                         DINOv2 patch-token std (1024), mean over tiles
                                       (texture statistics = diagonal of a Gram matrix)

Progress is saved in chunks; an interrupted run resumes where it stopped.

Local (Apple GPU, ~2 images/s):
    /opt/anaconda3/bin/python deep/embed.py --data-root data/wikiart5 --out outputs/deep/embeddings/emb_5.npz
Kaggle (GPU, see deep/kaggle_embed.md):
    python embed.py --data-root /kaggle/input --out /kaggle/working/emb_all.npz --batch-size 64
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import open_clip
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

CLASSES = ["Academic_Art", "Art_Nouveau", "Baroque", "Expressionism", "Japanese_Art",
           "Neoclassicism", "Primitivism", "Realism", "Renaissance", "Rococo",
           "Romanticism", "Symbolism", "Western_Medieval"]
OUTPUT_NAMES = ["clip_global", "clip_tiles", "dino_global", "dino_tiles", "dino_style"]
CHUNK = 2048
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
def find_images(data_root):
    """Walk data_root; the class is the first folder name in the path that is a known class.
    Works for both data/complete_data/Class/x.jpg and Kaggle's Class/Class/x.jpg layout."""
    found = {}
    for root, _, files in os.walk(data_root):
        cls = next((p for p in Path(root).parts if p in CLASSES), None)
        if cls:
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    found.setdefault(f"{cls}/{f}", os.path.join(root, f))
    rel_paths = sorted(found)
    return rel_paths, [found[r] for r in rel_paths]


class Views(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i])
            img.draft("RGB", (896, 896))          # fast JPEG decode at reduced size (keeps >= 448 px)
            img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (448, 448))    # unreadable file -> black image (reported below)
        g = TF.center_crop(TF.resize(img, 224, interpolation=TF.InterpolationMode.BICUBIC), 224)
        big = TF.center_crop(TF.resize(img, 448, interpolation=TF.InterpolationMode.BICUBIC), 448)
        tiles = [TF.crop(big, top, left, 224, 224) for top in (0, 224) for left in (0, 224)]
        return torch.stack([TF.to_tensor(v) for v in [g] + tiles])   # (5, 3, 224, 224) in [0, 1]


# ---------------------------------------------------------------- models
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(dev):
    clip, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
    clip = clip.visual.eval().half().to(dev)
    dino = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=224).eval().half().to(dev)
    return clip, dino


def normalise(x, mean, std):
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def embed_batch(views, clip, dino):
    """views: (B, 5, 3, 224, 224) -> dict of (B, D) tensors."""
    b = views.shape[0]
    x = views.flatten(0, 1)                                   # (B*5, 3, 224, 224)

    c = clip(normalise(x, CLIP_MEAN, CLIP_STD))
    c = torch.nn.functional.normalize(c.float(), dim=-1).view(b, 5, -1)

    tokens = dino.forward_features(normalise(x, IMNET_MEAN, IMNET_STD)).float()
    cls = tokens[:, 0].view(b, 5, -1)
    patch_std = tokens[:, dino.num_prefix_tokens:].std(dim=1).view(b, 5, -1)

    return {
        "clip_global": c[:, 0], "clip_tiles": c[:, 1:].mean(1),
        "dino_global": cls[:, 0], "dino_tiles": cls[:, 1:].mean(1),
        "dino_style": patch_std[:, 1:].mean(1),
    }


def run(files, clip, dino, dev, batch_size, workers):
    loader = DataLoader(Views(files), batch_size=batch_size, num_workers=workers)
    out = {k: [] for k in OUTPUT_NAMES}
    for views in loader:
        for k, v in embed_batch(views.to(dev).half(), clip, dino).items():
            out[k].append(v.cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True, help="output .npz file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bench", type=int, default=0, help="only time N images")
    args = parser.parse_args()

    dev = get_device()
    rel_paths, files = find_images(args.data_root)
    print(f"device={dev}  images={len(files)}", flush=True)
    clip, dino = load_models(dev)

    if args.bench:
        t = time.time()
        run(files[:args.bench], clip, dino, dev, args.batch_size, args.workers)
        rate = args.bench / (time.time() - t)
        print(f"{rate:.1f} images/s  ->  full run ≈ {len(files) / rate / 3600:.1f} h")
        return

    out = Path(args.out)
    chunk_dir = out.parent / f"{out.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(files) + CHUNK - 1) // CHUNK
    start = time.time()
    for ci in range(n_chunks):
        f = chunk_dir / f"chunk_{ci:03d}.npz"
        if not f.exists():
            np.savez(f, **run(files[ci * CHUNK:(ci + 1) * CHUNK], clip, dino, dev, args.batch_size, args.workers))
        print(f"chunk {ci + 1}/{n_chunks}  ({(time.time() - start) / 60:.1f} min)", flush=True)

    chunks = [np.load(chunk_dir / f"chunk_{ci:03d}.npz") for ci in range(n_chunks)]
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in OUTPUT_NAMES}
    np.savez(out, rel_paths=np.array(rel_paths), **arrays)
    print(f"Saved {out}  ({len(rel_paths)} images, {(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
