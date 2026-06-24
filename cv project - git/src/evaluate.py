"""
evaluate.py — Module 6 (Evaluation Dashboard) for the classifier
Runs the trained ResNet18 over the held-out TEST split and produces
report-ready artifacts in outputs/:
  - confusion_matrix.png
  - per_class_metrics.png  (precision / recall / F1 bars)
  - metrics_report.txt     (full sklearn classification report + accuracy)

Usage:
    python src/evaluate.py --data dataset/classification/test --weights models/fire_model.pth

(For the YOLO detection/segmentation models, ultralytics already writes
 PR-curves, confusion matrices and mAP plots into runs/ during .val().)
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from classifier import FireClassifier


def evaluate(data_dir, weights, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    clf = FireClassifier(weights)
    ds = datasets.ImageFolder(data_dir, transform=clf.transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    names = ds.classes

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            p = clf.model(x.to(clf.device)).argmax(1).cpu().numpy()
            y_pred.extend(p); y_true.extend(y.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=names, digits=4)
    with open(os.path.join(out_dir, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall accuracy: {acc*100:.2f}%\n\n{report}\n")
    print(f"Overall accuracy: {acc*100:.2f}%\n")
    print(report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # per-class precision / recall / F1
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(names)))
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(6, 4))
    plt.bar(x - w, prec, w, label="Precision", color="#f97316")
    plt.bar(x, rec, w, label="Recall", color="#60a5fa")
    plt.bar(x + w, f1, w, label="F1", color="#10b981")
    plt.xticks(x, names); plt.ylim(0, 1.05); plt.legend()
    plt.title("Per-class Metrics"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=150)
    plt.close()

    print(f"\nSaved confusion_matrix.png, per_class_metrics.png, metrics_report.txt to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/classification/test")
    ap.add_argument("--weights", default="models/fire_model.pth")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    evaluate(args.data, args.weights, args.out)
