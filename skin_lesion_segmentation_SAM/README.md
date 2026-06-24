# Skin Lesion Segmentation with SAM — SkinDisNet

Computer Vision course project (Week 4: Segmentation + Evaluation).
Zero-shot lesion segmentation on the **SkinDisNet** skin-disease dataset using
Meta AI's **Segment Anything Model (SAM)** — no training required.

## SAM zero-shot **and** a trained U-Net

SkinDisNet was published for **classification** and ships **without segmentation
masks**. With no ground truth, training a supervised segmenter from scratch would
require annotating thousands of images first. SAM is a powerful *pretrained, zero-shot*
segmenter, so we get lesion masks immediately and only hand-annotate a *small* subset
(~18 images) to compute evaluation metrics.

But zero-shot SAM is not trained on SkinDisNet, and its point prompt often grabs the
**whole hand/limb** instead of the lesion. So **Section 7** closes the loop **with no
human annotation**: SAM auto-labels the images, we **refine those labels to the diseased
patch** (color contrast), then **train a U-Net (ResNet-34) on SkinDisNet**, **save the
weights** (`unet_skindisnet.pt`), and use that trained model to segment automatically.
A visual **SAM (zero-shot) vs U-Net (trained)** grid is saved so you can see the trained
model focus on the lesion. (Hand-annotating ~18 images for IoU/Dice numbers stays
*optional* in Section 6 — it is not required to train or run the model.)

## Files

| File | Purpose |
|------|---------|
| `Skin_Lesion_Segmentation_SAM.ipynb` | Main notebook (run in Google Colab) |
| `_build_notebook.py` | Script that generated the notebook (regenerate with `python _build_notebook.py`) |
| `README.md` | This file |
| `PAPER_OUTLINE.md` | Section-by-section research-paper template |
| `VIDEO_SCRIPT.md` | Narration script for the video demo |

## How to run (Google Colab)

1. **Get the dataset** from Mendeley:
   <https://data.mendeley.com/datasets/yj3md44hxg/2> → download the zip.
2. Upload the zip to your Google Drive, e.g. `MyDrive/SkinDisNet/yj3md44hxg-2.zip`.
3. Open `Skin_Lesion_Segmentation_SAM.ipynb` in Colab
   (File → Upload notebook).
4. **Runtime → Change runtime type → GPU (T4)**.
5. Run cells top to bottom. In Section 3, edit `ZIP_PATH` / `DATA_ROOT` to match
   where you put the data.
6. (For rigorous evaluation) Hand-annotate ~15–20 images — see *Evaluation* below —
   and point `GT_DIR` / `GT_IMG_DIR` at them. If you skip this, the notebook uses an
   Otsu pseudo-ground-truth fallback so the metrics still compute.

## Pipeline

1. **Setup** — GPU check, install `segment-anything`, download SAM `vit_b` checkpoint.
2. **Data** — mount Drive, unzip, auto-detect the 6 class folders, visualize samples.
3. **SAM segmentation** — `SamAutomaticMaskGenerator` produces candidate masks; we
   pick the **lesion** by scoring each candidate on color-contrast vs. healthy skin
   (image border) and centrality. A center-point prompt is the fallback.
4. **Batch run** — segment N images/class, save masks + overlays + a predictions CSV.
5. **Evaluation** — IoU, Dice, Precision, Recall, Pixel Accuracy on the annotated
   subset; boxplot + qualitative grid figures saved for the paper.
6. **Train + save a U-Net** — auto-label with SAM (no annotation), refine labels to the
   lesion, train a U-Net on SkinDisNet, **save `unet_skindisnet.pt`**, and compare it
   visually against zero-shot SAM. The demo app then uses the trained model.
7. **Package** — zip all outputs (incl. trained weights) for submission.

## Evaluation: making the ground truth (hand-annotation guide)

**What this is:** SkinDisNet has no "correct answer" masks, so we make our own for a
small set. A **lesion** = the diseased patch of skin (the rash / discolored / infected
area). You outline that patch on ~18 images; those outlines become the ground truth the
notebook compares SAM against to compute IoU / Dice / etc.

**Do the whole thing in one sitting** — makesense.ai has no login and saves nothing;
if you close or refresh the tab, your work is gone.

### Step 1 — Export 18 images from Colab (auto-picks them for you)
You do **not** choose images by hand. Run this cell in the notebook; it picks 3 images
per class (~18 total, balanced) and downloads a zip:

```python
import os, shutil, glob
GT_IMG_DIR = '/content/drive/MyDrive/SkinDisNet/gt_images'
os.makedirs(GT_IMG_DIR, exist_ok=True)
pick = df.groupby('label', group_keys=False).apply(
    lambda g: g.sample(min(3, len(g)), random_state=99))
for _, r in pick.iterrows():
    shutil.copy(r['path'], os.path.join(GT_IMG_DIR, os.path.basename(r['path'])))
shutil.make_archive('/content/gt_images', 'zip', GT_IMG_DIR)
from google.colab import files
files.download('/content/gt_images.zip')
print('Exported', len(pick), 'images to', GT_IMG_DIR)
```

Unzip `gt_images.zip` anywhere on your PC (it does **not** need to be in the project folder).

### Step 2 — Outline the lesions at makesense.ai
1. Go to **https://www.makesense.ai** → **Get Started**.
2. Drag in all 18 images → choose **Object Detection** → **Start project**.
3. Create one label named **`lesion`** (a "label" is just a name tag for each shape you draw).
4. In the left toolbar pick the **Polygon** tool (pentagon icon). **Use Polygon, not Rectangle** —
   a rectangle includes healthy-skin corners and makes your scores look worse than they are.
5. For **each image**:
   - Click points all the way around the edge of the rash, then click the **first point again to close** the loop.
   - A dropdown appears → **select `lesion`**. *(Skipping this is the #1 mistake — an unlabeled
     shape is silently dropped on export, giving you 0 masks.)*
   - You should see a filled shape with "lesion" written on it.
6. **Before closing the tab:** **Actions → Export Annotations → Single file in COCO JSON format → Export.**
7. Upload that JSON to your Drive, e.g. `MyDrive/SkinDisNet/annotations.json`.

### Step 3 — Convert your outlines into mask PNGs (run in Colab)
Handles both polygons and rectangles. The first print line must show **both numbers > 0**:

```python
import json, os, numpy as np, cv2
from collections import defaultdict

COCO_JSON = '/content/drive/MyDrive/SkinDisNet/annotations.json'
GT_DIR    = '/content/drive/MyDrive/SkinDisNet/gt_masks'
os.makedirs(GT_DIR, exist_ok=True)

with open(COCO_JSON) as f:
    coco = json.load(f)
print(f"images={len(coco['images'])}  annotations={len(coco['annotations'])}")  # both must be > 0

images = {im['id']: im for im in coco['images']}
anns   = defaultdict(list)
for ann in coco['annotations']:
    anns[ann['image_id']].append(ann)

written = 0
for img_id, im in images.items():
    h, w = im['height'], im['width']
    mask = np.zeros((h, w), np.uint8)
    for ann in anns.get(img_id, []):
        seg = ann.get('segmentation')
        if seg:                                   # polygon (preferred)
            for poly in seg:
                pts = np.array(poly, np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)
        elif ann.get('bbox'):                     # rectangle fallback
            x, y, bw, bh = [int(v) for v in ann['bbox']]
            cv2.rectangle(mask, (x, y), (x+bw, y+bh), 255, -1)
    base = os.path.splitext(im['file_name'])[0]
    cv2.imwrite(os.path.join(GT_DIR, base + '.png'), mask)
    written += 1
print('Wrote', written, 'masks to', GT_DIR)
```

**Troubleshooting:**
- `Wrote 0 masks` / `images=0 annotations=0` → the export was empty: shapes weren't labeled
  `lesion`, or the tab was closed before exporting. Redo Step 2.
- `NameError: images not defined` → you ran only part of the cell; run the **whole** cell above.
- `FileNotFoundError` → `COCO_JSON` path is wrong. If you uploaded straight into Colab (left
  file panel, not Drive), use `/content/annotations.json`.

### Step 4 — Run the evaluation
The Section 6.1 cell already points at these folders:
```python
GT_DIR     = '/content/drive/MyDrive/SkinDisNet/gt_masks'
GT_IMG_DIR = '/content/drive/MyDrive/SkinDisNet/gt_images'
```
Run it — it should print **`Using hand-annotated GT: True`** — then run 6.2–6.4 for real metrics.

**Sanity check** a mask before trusting the numbers (should be a white blob on black):
```python
import matplotlib.pyplot as plt, glob, cv2
g = sorted(glob.glob('/content/drive/MyDrive/SkinDisNet/gt_masks/*.png'))[0]
plt.imshow(cv2.imread(g, 0), cmap='gray'); plt.title(g.split('/')[-1]); plt.show()
```

## Outputs (in `/content/outputs`, zipped at the end)

- `masks/` — predicted binary masks (PNG)
- `overlays/` — image + mask overlay (PNG, great for the paper/video)
- `predictions.csv` — per-image mask coverage
- `metrics_per_image.csv`, `metrics_summary.csv` — evaluation numbers
- `metrics_boxplot.png`, `qualitative_grid.png` — paper figures

## Tuning

- Better masks: set `SAM_TYPE = 'vit_l'` or `'vit_h'` (bigger download, slower, sharper).
- More coverage: raise `PER_CLASS` in Section 5.
- Lesion selection too aggressive/conservative: adjust the thresholds in
  `lesion_score()` (area fraction bounds) and the generator params.

## Honest limitations (state these in the paper)

- No official ground-truth masks → evaluation is on a *small* hand-annotated set.
- The lesion-selection heuristic is unsupervised; it can fail on images with multiple
  lesions or strong lighting/hair artifacts.
- Augmented images are near-duplicates of originals; evaluate on **original** images
  to avoid optimistic bias.
