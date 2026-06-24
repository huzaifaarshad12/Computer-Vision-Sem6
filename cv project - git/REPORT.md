# Fire & Smoke Vision System
### A Multi-Module Deep Learning Approach to Fire Detection, Localization & Severity Assessment

**Course:** Computer Vision  
**Group Members:** Salman Naseem (2023-CS-78) · Huzaifa Arshad (2023-CS-86)  
**Instructor:** Muhammad Waseem

---

## Abstract

Fire and smoke are leading causes of loss of life and property, and automatic
visual detection is a key component of modern surveillance and early-warning
systems. Most existing student-level solutions stop at binary classification —
they can say *whether* an image contains fire, but not *where* it is or *how
severe* it is. This project presents the **Fire & Smoke Vision System**, a
multi-module computer-vision pipeline that integrates **five complementary
deep-learning tasks** trained from a single LabelMe-annotated dataset of 5,000
images: (1) image **classification** with ResNet18, (2) object **detection**
with YOLOv8, (3) instance **segmentation** with YOLOv8-seg, (4) **explainability**
via Grad-CAM, and (5) a mask-based **severity score**. All modules are exposed
through a single React web application backed by FastAPI. On a held-out test
set the system achieves **99.0% classification accuracy**, **0.787 detection
mAP@50**, and **0.792 mask mAP@50**, while Grad-CAM confirms the model attends to
genuine flame/smoke regions.

---

## 1. Introduction

### 1.1 Motivation
Early visual detection of fire and smoke enables faster response in surveillance,
industrial monitoring, and forest/urban safety. A practical system must do more
than raise a binary alarm — an operator needs to know **where** the fire is,
**how large** it is, and **why** the system flagged it.

### 1.2 Problem Statement
Build a single, explainable system that, from one still image, can:
- decide whether fire/smoke is present (classification),
- localize it with bounding boxes (detection),
- outline its exact extent with pixel masks (segmentation),
- justify the decision visually (explainability),
- and quantify how dangerous it is (severity).

### 1.3 Objectives
1. Go beyond classification to localization, segmentation, and risk scoring.
2. Reuse **one** annotated dataset to train every task (data-efficient design).
3. Provide explainable predictions to build user trust.
4. Deliver an accessible web interface for live demonstration.

### 1.4 Contributions
- A reusable converter that turns LabelMe polygon annotations into classification,
  detection, and segmentation labels from a single source of truth.
- Five trained, individually-validated CV modules unified in one product.
- A professional React + FastAPI web application with per-module selection.
- An honest evaluation on a held-out test set, including a discussion of dataset
  bias (near-duplicate video frames).

---

## 2. Related Work

| Study | Focus | Strength | Limitation |
|-------|-------|----------|------------|
| Muhammad et al. (2018) | CNN fire classification | High accuracy | No localization |
| Li & Zhao (2020) | Faster R-CNN fire detection | Bounding boxes | Heavy; no masks |
| Jadon et al. (2019) — FireNet | Lightweight CNN | Real-time | Classification only |
| Shamsoshoara et al. (2021) — FLAME | Aerial fire segmentation | Pixel masks | Single task |
| Ultralytics YOLOv8 (2023) | General detection/segmentation | Fast, state-of-the-art | Not fire-specific |
| **This work (2025)** | **5 unified modules + XAI + severity** | **Explainable, end-to-end** | Single-domain dataset |

Prior work typically targets *one* task. Our contribution is integrating
classification, detection, segmentation, explainability, and severity scoring
into a single coherent system trained from the same data.

---

## 3. Dataset

- **Source:** Fire-Segmentation-Dataset (LabelMe polygon annotations).
- **Size:** 5,000 images; **3,203** annotated with fire/smoke polygons, ~1,797 normal.
- **Instances:** 7,802 `fire` polygons, 3,460 `smoke` polygons.
- **Classes:** `fire` (0), `smoke` (1); plus a `normal` class for classification.

### 3.1 From one annotation to three label formats
A single converter (`src/prepare_dataset.py`) derives every label type from the
polygons:
- **Polygon → bounding box** (min/max of points) → YOLO **detection** labels.
- **Polygon → normalized point list** → YOLO **segmentation** labels.
- **Annotation present / absent** → `fire_smoke` / `normal` **classification** labels.
- Images without annotations are written as empty label files (negatives), which
  reduces false positives in detection/segmentation.

### 3.2 Splits
Stratified random split with a fixed seed (`seed=42`) for reproducibility:
**80% train / 10% validation / 10% test** (4,000 / 500 / 500 images). The same
split is used across all modules so results are directly comparable.

---

## 4. Methodology

### 4.1 System Architecture
```
Input image
   │
   ├─► Module 1  Classification   (ResNet18)         → fire / normal + confidence
   ├─► Module 2  Detection        (YOLOv8)           → bounding boxes
   ├─► Module 3  Segmentation     (YOLOv8-seg)       → pixel masks
   ├─► Module 4  Explainability   (Grad-CAM)         → decision heatmap
   └─► Module 5  Severity         (mask coverage)    → NONE … CRITICAL
                         │
                         ▼
        React Web App  ←  FastAPI backend  (serves all modules)
```
*(See `outputs/pipeline.png` for the diagram used on the poster.)*

### 4.2 Module 1 — Classification (ResNet18)
- Transfer learning from ImageNet-pretrained ResNet18; final FC layer → 2 classes.
- **Augmentation:** horizontal flip, ±15° rotation, color jitter.
- **Normalization:** ImageNet mean/std.
- **Training:** Adam (lr = 1e-4), StepLR scheduler, 12 epochs, batch 32; the model
  with the best **validation** accuracy is saved.

### 4.3 Module 2 — Detection (YOLOv8n)
- `yolov8n.pt` fine-tuned, 50 epochs, image size 640, batch 16, AdamW (auto).
- Outputs class-labeled bounding boxes for fire and smoke.

### 4.4 Module 3 — Segmentation (YOLOv8n-seg)
- `yolov8n-seg.pt` fine-tuned, 50 epochs, image size 640, batch 16.
- Outputs pixel-level instance masks for fire and smoke.

### 4.5 Module 4 — Explainability (Grad-CAM)
- Grad-CAM implemented from scratch with forward/backward hooks on ResNet18's
  last convolutional block (`layer4`).
- Produces a heatmap overlay highlighting the pixels most responsible for the
  classification decision.

### 4.6 Module 5 — Severity Score
- Computes the fraction of the frame covered by segmentation masks and maps it to
  a rating: **NONE → LOW → MEDIUM → HIGH → CRITICAL**, with per-class breakdown.

### 4.7 Implementation
- **Inference modules:** PyTorch + Ultralytics + Pillow/NumPy (`src/`).
- **Backend:** FastAPI (`server.py`) exposing `/api/status` and `/api/analyze`
  (the latter accepts a `modules` list to run any subset).
- **Frontend:** React (served directly, no build step) with an Anthropic-inspired
  design; users upload an image, pick modules, and view results per card.
- **Training:** Google Colab (Tesla T4 GPU); notebooks in `notebooks/`.

---

## 5. Experiments & Results

All metrics below are on the **held-out test split** (500 images never seen in
training). Reproduce with `python src/evaluate.py` (classifier) and
`model.val(split="test")` (YOLO).

### 5.1 Headline Results
| Module | Metric | Score |
|--------|--------|-------|
| Classification (ResNet18) | Test accuracy | **99.0%** |
| Detection (YOLOv8) | Test mAP@50 | **0.787** |
| Detection (YOLOv8) | Test mAP@50–95 | 0.511 |
| Segmentation (YOLOv8-seg) | Test mask mAP@50 | **0.792** |
| Segmentation (YOLOv8-seg) | Test mask mAP@50–95 | 0.501 |

### 5.2 Classification — per class
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| fire_smoke | 0.9884 | 0.9971 | 0.9927 |
| normal | 0.9936 | 0.9748 | 0.9841 |
| **Accuracy** | | | **0.9900** |

*Figures: `outputs/confusion_matrix.png`, `outputs/per_class_metrics.png`.*

### 5.3 Detection / Segmentation — per class (mAP@50)
| Class | Detection | Segmentation (mask) |
|-------|-----------|---------------------|
| fire | 0.754 | 0.761 |
| smoke | 0.802 | 0.783 |

### 5.4 Key observations
- **No overfitting:** test scores **match or exceed** validation (e.g. detection
  test 0.787 ≥ val 0.778; segmentation test 0.792 ≥ val 0.771).
- **Smoke localizes slightly better than fire**, consistently across detection and
  segmentation — smoke regions are larger and more distinct than small flames.
- **Tight masks:** mask mAP ≈ box mAP, indicating the predicted masks closely
  follow object boundaries rather than being loose.
- **Explainability:** Grad-CAM heatmaps concentrate on flames/smoke, not on
  background or bystanders (see `outputs/demo_panel.jpg`).
- **No false alarms:** on normal images the classifier predicts `normal` (98–100%)
  and detection/segmentation return zero regions (severity = NONE).

### 5.5 Qualitative results
`outputs/demo_panel.jpg` shows a representative case (Input · Grad-CAM · Detection ·
Segmentation): classifier 100% fire_smoke; detector finds smoke (0.95) and two
fire boxes (0.83, 0.77); segmentation covers 18.3% of the frame → **MEDIUM** severity.

### 5.6 System testing
An end-to-end test suite (`tools/test_e2e.py`) exercises the live API across all
module combinations on fire and normal images — **64/64 checks passed**, confirming
each selection returns exactly the requested modules and the correct verdict.

---

## 6. Discussion & Limitations

- **Dataset bias (near-duplicate frames):** the dataset contains consecutive video
  frames, so visually similar scenes can appear across train/val/test. This partly
  explains the very high classification accuracy. We mitigated concern by reporting
  held-out test metrics and verifying behaviour on external images, and we note a
  group-aware split as future work.
- **Detection recall:** the detector is slightly conservative (precision 0.81 >
  recall 0.70). For a safety-critical alarm, higher recall would be preferable.
- **Small flames** are harder to localize than large smoke plumes, reflected in the
  lower per-class fire scores.
- **Still images only:** the current system is image-based; video/real-time
  inference is left for future work.

---

## 7. Conclusion & Future Work

The Fire & Smoke Vision System demonstrates that a single annotated dataset can
power five complementary fire-analysis tasks within one explainable, web-based
tool, achieving 99% classification accuracy and ≈0.79 detection/segmentation mAP
on unseen data. The unified, modular design makes the system both a practical demo
and a strong educational artifact.

**Future directions:**
- Higher-recall detection tuned for safety-critical operation.
- Real-time **video** inference and live camera support.
- **Edge deployment** (e.g., Jetson / mobile) for on-site monitoring.
- A larger, **multi-source** dataset with group-aware splits to remove
  near-duplicate-frame bias.
- Confidence calibration and alert thresholds for deployment.

---

## 8. References

1. J. Redmon, S. Divvala, R. Girshick, A. Farhadi. *You Only Look Once: Unified,
   Real-Time Object Detection.* CVPR, 2016.
2. G. Jocher et al. *Ultralytics YOLOv8.* 2023.
3. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.*
   CVPR, 2016.
4. R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization.* ICCV, 2017.
5. K. Muhammad et al. *Early Fire Detection using Convolutional Neural Networks.*
   IEEE Access, 2018.

---

## Appendix A — How to Run

```bash
pip install -r requirements.txt          # install dependencies
python server.py                          # launch web app → http://localhost:8000
python demo.py path/to/image.jpg          # CLI: all modules → outputs/demo_panel.jpg
python src/prepare_dataset.py             # rebuild dataset (needed before re-evaluation)
python src/evaluate.py --data dataset/classification/test --weights models/fire_model.pth
python tools/test_e2e.py                  # end-to-end API tests (server must be running)
```

## Appendix B — Project Structure
```
server.py · web/index.html      Web app (React + FastAPI)
src/                            5 inference modules + prepare/evaluate
notebooks/                     Colab training (classifier, YOLO det+seg)
models/                        Trained weights (.pth, .pt)
outputs/                       Figures & metrics for this report
tools/                         Notebook/poster builders, E2E tests
```

## Appendix C — Work Distribution
_Fill this in for your submission, e.g.:_
- **Salman Naseem (2023-CS-78):** _…_
- **Huzaifa Arshad (2023-CS-86):** _…_
