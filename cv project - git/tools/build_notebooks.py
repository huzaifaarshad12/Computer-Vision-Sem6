"""
build_notebooks.py
==================
Generates the Colab training notebooks under notebooks/.
Kept as a tool so the embedded dataset-prep code stays in sync with
src/prepare_dataset.py (DRY). Run:  python tools/build_notebooks.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

with open(os.path.join(ROOT, "src", "prepare_dataset.py"), "r", encoding="utf-8") as f:
    PREP_CODE = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


def _lines(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def save(name, cells):
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


# Shared cells -------------------------------------------------------------
DATASET_ZIP = "/content/drive/MyDrive/Fire-Segmentation-Dataset-main.zip"

mount_unzip = code(f"""
# === Mount Google Drive & unzip the dataset =================================
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
DATASET_ZIP = "{DATASET_ZIP}"   # <-- change if your zip lives elsewhere
EXTRACT = "/content/fire_src"
with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
    z.extractall(EXTRACT)

# locate the folder that actually contains images/ and json/
IMAGES = JSON = None
for root, dirs, files in os.walk(EXTRACT):
    if root.endswith("images"):
        IMAGES = root
    if root.endswith("json"):
        JSON = root
print("images:", IMAGES)
print("json  :", JSON)
""")

writefile_prep = code("%%writefile prepare_dataset.py\n" + PREP_CODE)

run_prep = code("""
# === Convert LabelMe polygons -> classification + YOLO det + YOLO seg ========
from prepare_dataset import prepare
prepare(IMAGES, JSON, out_dir="/content/dataset")
""")


# ===========================================================================
# Notebook 1 : Classifier
# ===========================================================================
classifier_cells = [
    md("""
# 🔥 Module 1 — Fire/Smoke Classifier (improved ResNet18)

Trains a ResNet18 to classify an image as **fire_smoke** vs **normal**, but unlike the
first version it adds: a proper **train/val/test split**, **data augmentation**,
**ImageNet normalization**, **validation tracking** (saves the *best* model), and a
**confusion matrix + classification report** on a held-out test set.

> Runtime → Change runtime type → **GPU** before running.
"""),
    code("!pip -q install scikit-learn seaborn matplotlib"),
    mount_unzip,
    writefile_prep,
    run_prep,
    code("""
# === Data loaders with augmentation =========================================
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

ROOT = "/content/dataset/classification"
train_ds = datasets.ImageFolder(f"{ROOT}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{ROOT}/val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(f"{ROOT}/test",  transform=eval_tf)
print("classes:", train_ds.classes)
print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)
"""),
    code("""
# === Model + training with validation tracking ==============================
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 12
best_val = 0.0

def accuracy(loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            correct += (p == y).sum().item(); total += y.size(0)
    return correct / total * 100

for epoch in range(EPOCHS):
    model.train(); running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        running += loss.item()
    scheduler.step()
    val_acc = accuracy(val_loader)
    print(f"Epoch {epoch+1:2d}: loss={running/len(train_loader):.4f}  val_acc={val_acc:.2f}%")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "fire_model.pth")
        print(f"   ↳ saved new best ({best_val:.2f}%)")

print("Best val accuracy:", best_val)
"""),
    code("""
# === Evaluate on the held-out TEST set: report + confusion matrix ===========
import numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

model.load_state_dict(torch.load("fire_model.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        y_pred.extend(p); y_true.extend(y.numpy())

names = test_ds.classes
print(classification_report(y_true, y_pred, target_names=names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=names, yticklabels=names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Classifier Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.show()
"""),
    code("""
# === Save model to Drive + download =========================================
import shutil
shutil.copy("fire_model.pth", "/content/drive/MyDrive/fire_model.pth")
from google.colab import files
files.download("fire_model.pth")
files.download("confusion_matrix.png")
print("Put fire_model.pth into your project's  models/  folder.")
"""),
]


# ===========================================================================
# Notebook 2 : YOLO detection + segmentation
# ===========================================================================
yolo_cells = [
    md("""
# 🎯 Modules 2 & 3 — YOLOv8 Detection + Segmentation

Trains two YOLOv8 models on the SAME prepared dataset:
* **Detection** → bounding boxes around fire / smoke
* **Segmentation** → pixel-level masks (also powers the *severity score* in the app)

> Runtime → Change runtime type → **GPU** before running.
"""),
    code("!pip -q install ultralytics"),
    mount_unzip,
    writefile_prep,
    run_prep,
    code("""
# === Module 2: Train YOLOv8 DETECTION ======================================
from ultralytics import YOLO
det = YOLO("yolov8n.pt")          # nano = fast; use yolov8s.pt for more accuracy
det.train(data="/content/dataset/data_det.yaml",
          epochs=50, imgsz=640, batch=16, name="fire_detect", patience=10)
metrics = det.val()
print("Detection mAP50:", metrics.box.map50)
"""),
    code("""
# === Save DETECTION weights -> Drive + download (run this before segmentation!)
import shutil
det_w = "/content/runs/detect/fire_detect/weights/best.pt"
shutil.copy(det_w, "fire_detect.pt")
shutil.copy("fire_detect.pt", "/content/drive/MyDrive/fire_detect.pt")
from google.colab import files
files.download("fire_detect.pt")
print("✅ Detection saved to Drive + downloaded. Put fire_detect.pt in models/.")
"""),
    code("""
# === Module 3: Train YOLOv8 SEGMENTATION ===================================
# OPTIONAL / can be run in a separate session. The app works fine without it
# (the Segment tab just shows a hint until this weight exists).
# NOTE: re-run the install + mount + writefile + prepare cells above first if
# this is a fresh Colab session.
from ultralytics import YOLO
seg = YOLO("yolov8n-seg.pt")
seg.train(data="/content/dataset/data_seg.yaml",
          epochs=50, imgsz=640, batch=16, name="fire_seg", patience=10)
metrics = seg.val()
print("Segmentation mask mAP50:", metrics.seg.map50)
"""),
    code("""
# === Save SEGMENTATION weights -> Drive + download (run after seg training) =
import shutil
seg_w = "/content/runs/segment/fire_seg/weights/best.pt"
shutil.copy(seg_w, "fire_seg.pt")
shutil.copy("fire_seg.pt", "/content/drive/MyDrive/fire_seg.pt")
from google.colab import files
files.download("fire_seg.pt")
print("✅ Segmentation saved to Drive + downloaded. Put fire_seg.pt in models/.")
"""),
]

save("01_train_classifier.ipynb", classifier_cells)
save("02_train_yolo.ipynb", yolo_cells)
print("\nDone. Two notebooks generated in notebooks/.")
