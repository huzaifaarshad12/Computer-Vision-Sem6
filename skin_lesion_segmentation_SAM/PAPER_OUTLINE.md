# Research Paper — Outline & Template

**Working title:** *Zero-Shot Skin Lesion Segmentation on Clinical Images Using the
Segment Anything Model (SAM): A Study on the SkinDisNet Dataset*

Fill the **[brackets]** with your numbers after running the notebook. Target length
4–6 pages (IEEE two-column or your course template).

---

## Abstract (~150–200 words)
- Problem: automatic skin-lesion segmentation on clinical photos.
- Gap: SkinDisNet has 6 disease classes but **no segmentation masks**.
- Method: apply **SAM** (pretrained, zero-shot) with an unsupervised lesion-selection
  heuristic (color contrast + centrality); evaluate on a hand-annotated subset.
- Results: mean IoU **[X.XX]**, Dice **[X.XX]** over **[N]** images.
- Takeaway: foundation models segment lesions with no task-specific training.

## 1. Introduction
- Why lesion segmentation matters (diagnosis support, lesion area/border analysis).
- SkinDisNet dataset and its classification-only design (no masks).
- Contribution bullets:
  1. A training-free segmentation pipeline for SkinDisNet using SAM.
  2. An unsupervised heuristic to pick the lesion among SAM's candidate masks.
  3. A quantitative evaluation protocol on a hand-annotated subset.

## 2. Related Work
- Classical segmentation (thresholding/Otsu, K-means, GrabCut, active contours).
- Deep segmentation (U-Net, DeepLab) and the **mask-scarcity** problem in dermatology.
- Foundation models / **SAM** for medical imaging (zero-shot, promptable).

## 3. Dataset
- SkinDisNet: 6 classes (Atopic Dermatitis, Contact Dermatitis, Eczema, Scabies,
  Seborrheic Dermatitis, Tinea Corporis), 512×512 JPG, 1,710 original + 11,970
  augmented, 416 patients, smartphone-captured at two Bangladeshi hospitals.
- Note the absence of segmentation ground truth → motivates the zero-shot approach.
- Table 1: image counts per class (from the notebook's value_counts).

## 4. Methodology
- **4.1 Overview** — figure of the pipeline (Image → SAM candidates → lesion
  selection → mask). Reuse the notebook's overlay outputs.
- **4.2 SAM** — ViT-B image encoder, automatic mask generator settings
  (points_per_side, IoU/stability thresholds, min region area).
- **4.3 Lesion selection heuristic** — score = color-contrast(region vs. border skin)
  × (0.5 + centrality); reject masks <1% or >85% of the frame; center-point prompt
  fallback. Give the equation.
- **4.4 Evaluation protocol** — hand-annotated subset of **[N]** images; metrics:
  IoU, Dice, Precision, Recall, Pixel Accuracy (define each with formulas).

## 5. Experiments & Results
- Setup: Google Colab, T4 GPU, SAM `vit_b`.
- **Table 2:** mean ± std for IoU/Dice/Precision/Recall/PixelAcc (from
  `metrics_summary.csv`).
- **Figure (boxplot):** `metrics_boxplot.png`.
- **Figure (qualitative grid):** `qualitative_grid.png` (Image | GT | Prediction).
- Optional per-class breakdown of mean IoU.
- Brief discussion of typical successes vs. failure cases (hair, lighting, multiple
  lesions).

## 6. Discussion / Limitations
- Small evaluation set; pseudo-GT fallback vs. hand-annotation.
- Heuristic is unsupervised and can mis-pick on cluttered images.
- Augmented images are near-duplicates — evaluate on originals.

## 7. Conclusion & Future Work
- SAM gives strong zero-shot lesion masks with no training.
- Future: prompt SAM with the classifier's attention; larger annotated test set;
  compare against a U-Net trained on SAM pseudo-labels.

## References
- SkinDisNet data paper (your ScienceDirect link, S2352340925009606).
- Kirillov et al., *Segment Anything*, ICCV 2023.
- Ronneberger et al., *U-Net*, MICCAI 2015.
- Otsu, *A threshold selection method*, 1979.
- Add 3–5 SAM-in-medical-imaging papers.

---

### Metric definitions to paste into 4.4
- IoU = |P ∩ G| / |P ∪ G|
- Dice = 2|P ∩ G| / (|P| + |G|)
- Precision = TP / (TP + FP), Recall = TP / (TP + FN)
- Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)

where P = predicted lesion pixels, G = ground-truth lesion pixels.
