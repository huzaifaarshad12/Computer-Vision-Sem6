# Results & Validation

All numbers below were measured on a **held-out test split** (10% of the data,
500 images, never seen during training). The split is reproducible with
`seed=42` in `src/prepare_dataset.py`, so these metrics can be regenerated.

## Module 1 — Classification (ResNet18)

Evaluated with `python src/evaluate.py --data dataset/classification/test --weights models/fire_model.pth`

| Metric | fire_smoke | normal | Overall |
|--------|-----------|--------|---------|
| Precision | 0.9884 | 0.9936 | — |
| Recall | 0.9971 | 0.9748 | — |
| F1 | 0.9927 | 0.9841 | — |
| **Accuracy** | | | **99.00%** |

Artifacts: `outputs/confusion_matrix.png`, `outputs/per_class_metrics.png`,
`outputs/metrics_report.txt`.

> Note: accuracy is high partly because the dataset contains consecutive video
> frames, so train/val/test can share visually similar scenes. We additionally
> confirmed correct behaviour on normal images (no false alarms) and via
> Grad-CAM, which shows the model attends to the actual fire region.

## Module 2 — Detection (YOLOv8n)

50 epochs, imgsz 640. Validated on the test split with `model.val(split="test")`.

| Metric | Value |
|--------|-------|
| mAP50 (test, all) | **0.787** |
| mAP50 (val, all) | 0.778 |
| mAP50-95 (test) | 0.511 |
| Precision | 0.808 |
| Recall | 0.699 |

Per-class (val) mAP50:  🔥 fire = 0.754  ·  💨 smoke = 0.802

Test mAP (0.787) ≥ val mAP (0.778) → the detector generalizes well, no overfit.
Ultralytics plots in `runs/detect/fire_detect/`: `results.png`, `PR_curve.png`,
`confusion_matrix.png`, `val_batch*_pred.jpg`.

## Module 3 — Segmentation (YOLOv8n-seg)

50 epochs, imgsz 640, ~1.4 h on a Tesla T4. Validated on the val split.

| Metric | Val (all) | Test (all) |
|--------|-----------|------------|
| **Mask mAP50** | 0.771 | **0.792** |
| Mask mAP50-95 | 0.474 | 0.501 |
| Box mAP50 | 0.781 | 0.797 |

Per-class (val) mask mAP50:  🔥 fire = 0.761  ·  💨 smoke = 0.783

Test mask mAP (0.792) ≥ val (0.771) → generalizes well, no overfit.
Mask mAP ≈ box mAP → the predicted masks are tight, not loose.
Plots in `runs/segment/fire_seg/`: `results.png`, `MaskPR_curve.png`,
`val_batch*_pred.jpg`. Drop `fire_seg.pt` into `models/` to activate the app's
Segment tab + severity score.

## Module 4 — Explainability (Grad-CAM)

Heatmaps confirm the classifier focuses on flame/smoke pixels rather than
background. Example: `outputs/demo_panel.jpg`.

## Module 5 — Severity score

Derived from the fraction of the frame covered by segmentation masks →
NONE / LOW / MEDIUM / HIGH / CRITICAL (`src/severity.py`). Active once
`fire_seg.pt` is present.

## End-to-end demo

`python demo.py <image>` runs all available modules and saves a combined panel
to `outputs/demo_panel.jpg` (Input · Grad-CAM · Detection · Segmentation).

### Validated sample (image 1000.jpg)
```
Classify : fire_smoke (100.0%)
Detect   : smoke 95%, fire 83%, fire 73%
```
Normal images: classifier → normal (98–100%), detector → 0 detections (no false alarms).
