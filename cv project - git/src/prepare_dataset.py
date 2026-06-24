"""
prepare_dataset.py
==================
Converts the LabelMe polygon dataset (Fire-Segmentation-Dataset) into THREE
ready-to-train dataset formats from a single source of truth:

  1. Classification  ->  classification/{train,val,test}/{fire_smoke,normal}/
  2. YOLO detection  ->  yolo_det/{images,labels}/{train,val,test}/  + data_det.yaml
  3. YOLO segmentation -> yolo_seg/{images,labels}/{train,val,test}/ + data_seg.yaml

Each LabelMe JSON contains polygon shapes labelled "fire" or "smoke".
  - A polygon -> bounding box (min/max of points)         => detection label
  - A polygon -> normalized point list                    => segmentation label
  - Any annotation present -> "fire_smoke", else "normal"  => classification label

Class IDs (kept consistent with data.yaml):   0 = fire,  1 = smoke

Run it directly:
    python src/prepare_dataset.py \
        --images Fire-Segmentation-Dataset-main/images \
        --json   Fire-Segmentation-Dataset-main/json \
        --out    dataset

Works the same on a local machine and on Google Colab.
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter

# fire = 0, smoke = 1  (matches data.yaml)
CLASS_MAP = {"fire": 0, "smoke": 1}
CLASS_NAMES = ["fire", "smoke"]


def polygon_to_bbox(points):
    """Polygon points [[x,y],...] -> (xmin, ymin, xmax, ymax) in pixels."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def clamp01(v):
    return max(0.0, min(1.0, v))


def load_shapes(json_path):
    """Return (shapes, width, height) from a LabelMe json, or (None, None, None)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    shapes = [s for s in data.get("shapes", []) if s.get("label") in CLASS_MAP]
    return shapes, w, h


def write_det_label(shapes, w, h, out_path):
    """YOLO detection label: one line per box -> 'cls xc yc bw bh' (normalized)."""
    lines = []
    for s in shapes:
        pts = s["points"]
        if len(pts) < 3:
            continue
        xmin, ymin, xmax, ymax = polygon_to_bbox(pts)
        xc = clamp01(((xmin + xmax) / 2) / w)
        yc = clamp01(((ymin + ymax) / 2) / h)
        bw = clamp01((xmax - xmin) / w)
        bh = clamp01((ymax - ymin) / h)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{CLASS_MAP[s['label']]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_seg_label(shapes, w, h, out_path):
    """YOLO segmentation label: one line per polygon -> 'cls x1 y1 x2 y2 ...' (normalized)."""
    lines = []
    for s in shapes:
        pts = s["points"]
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.append(f"{clamp01(x / w):.6f}")
            coords.append(f"{clamp01(y / h):.6f}")
        lines.append(f"{CLASS_MAP[s['label']]} " + " ".join(coords))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def prepare(images_dir, json_dir, out_dir, splits=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    split_of = {}
    for i, f in enumerate(image_files):
        if i < n_train:
            split_of[f] = "train"
        elif i < n_train + n_val:
            split_of[f] = "val"
        else:
            split_of[f] = "test"

    # --- create folder tree ---
    cls_root = os.path.join(out_dir, "classification")
    det_root = os.path.join(out_dir, "yolo_det")
    seg_root = os.path.join(out_dir, "yolo_seg")
    for split in ("train", "val", "test"):
        for cls in ("fire_smoke", "normal"):
            os.makedirs(os.path.join(cls_root, split, cls), exist_ok=True)
        for root in (det_root, seg_root):
            os.makedirs(os.path.join(root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    stats = Counter()
    for img in image_files:
        split = split_of[img]
        stem = img.rsplit(".", 1)[0]
        json_path = os.path.join(json_dir, stem + ".json")
        src_img = os.path.join(images_dir, img)

        shapes, w, h = (None, None, None)
        if os.path.exists(json_path):
            shapes, w, h = load_shapes(json_path)

        has_fire = bool(shapes) and w and h
        cls_name = "fire_smoke" if has_fire else "normal"
        stats[f"{split}/{cls_name}"] += 1

        # 1) classification: copy image into class folder
        shutil.copy(src_img, os.path.join(cls_root, split, cls_name, img))

        # 2) + 3) detection & segmentation: copy image + write labels
        for root, writer in ((det_root, write_det_label), (seg_root, write_seg_label)):
            shutil.copy(src_img, os.path.join(root, "images", split, img))
            label_path = os.path.join(root, "labels", split, stem + ".txt")
            if has_fire:
                writer(shapes, w, h, label_path)
            else:
                open(label_path, "w").close()  # empty = background/negative

    # --- write YOLO data yamls ---
    _write_yaml(os.path.join(out_dir, "data_det.yaml"), det_root)
    _write_yaml(os.path.join(out_dir, "data_seg.yaml"), seg_root)

    print("\n===== DATASET PREPARED =====")
    print(f"Total images: {n}")
    for k in sorted(stats):
        print(f"  {k:>18}: {stats[k]}")
    print(f"\nOutputs in: {os.path.abspath(out_dir)}")
    print("  - classification/   (ImageFolder format)")
    print("  - yolo_det/  + data_det.yaml")
    print("  - yolo_seg/  + data_seg.yaml")
    return stats


def _write_yaml(path, root):
    root_abs = os.path.abspath(root).replace("\\", "/")
    content = (
        f"path: {root_abs}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: fire\n"
        f"  1: smoke\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare fire/smoke datasets from LabelMe JSON")
    ap.add_argument("--images", default="Fire-Segmentation-Dataset-main/images")
    ap.add_argument("--json", default="Fire-Segmentation-Dataset-main/json")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare(args.images, args.json, args.out, seed=args.seed)
