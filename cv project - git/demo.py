"""
demo.py — run the whole pipeline on a single image and save a result panel.

Runs every module that has a trained model available (classification, Grad-CAM,
detection, and — if fire_seg.pt exists — segmentation + severity), then stitches
the outputs into one annotated image: outputs/demo_panel.jpg.

Usage:
    python demo.py                       # uses a sample image from the dataset
    python demo.py path/to/image.jpg
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def _find(*c):
    for p in c:
        fp = os.path.join(HERE, p)
        if os.path.exists(fp):
            return fp
    return None


CLASSIFIER_W = _find("models/fire_model.pth", "fire_model.pth")
DETECTOR_W = _find("models/fire_detect.pt")
SEGMENTER_W = _find("models/fire_seg.pt")

PANEL = 320


def _label(img, caption):
    """Resize to a square tile and write a caption bar at the top."""
    img = img.convert("RGB").resize((PANEL, PANEL))
    tile = Image.new("RGB", (PANEL, PANEL + 28), "#0b0f19")
    tile.paste(img, (0, 28))
    d = ImageDraw.Draw(tile)
    d.text((6, 7), caption, fill="#f8fafc")
    return tile


def main(image_path):
    tiles, summary = [], []

    original = Image.open(image_path)
    tiles.append(_label(original, "Input"))

    # Module 1 — classification
    if CLASSIFIER_W:
        from classifier import FireClassifier
        c = FireClassifier(CLASSIFIER_W).predict(image_path)
        summary.append(f"Classify : {c['label']} ({c['confidence']*100:.1f}%)")

    # Module 4 — Grad-CAM
    if CLASSIFIER_W:
        from gradcam import GradCAM
        g = GradCAM(CLASSIFIER_W).generate(image_path)
        tiles.append(_label(g["overlay"], f"Grad-CAM: {g['label']}"))

    # Module 2 — detection
    if DETECTOR_W:
        from detector import FireDetector
        d = FireDetector(DETECTOR_W).predict(image_path)
        tiles.append(_label(d["annotated"], f"Detect: {d['count']} object(s)"))
        summary.append("Detect   : " + (", ".join(
            f"{x['label']} {x['confidence']*100:.0f}%" for x in d["detections"]) or "none"))
    else:
        summary.append("Detect   : (no model)")

    # Module 3 + 5 — segmentation + severity
    if SEGMENTER_W:
        from segmenter import FireSegmenter
        s = FireSegmenter(SEGMENTER_W).predict(image_path)
        tiles.append(_label(s["annotated"], f"Segment: {s['severity']['level']}"))
        summary.append(f"Severity : {s['severity']['level']} ({s['severity']['percent']}% of frame)")
    else:
        summary.append("Severity : (no segmentation model yet)")

    # stitch tiles into a row
    w = PANEL * len(tiles)
    panel = Image.new("RGB", (w, PANEL + 28), "#0b0f19")
    for i, t in enumerate(tiles):
        panel.paste(t, (i * PANEL, 0))
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    out = os.path.join(HERE, "outputs", "demo_panel.jpg")
    panel.save(out)

    print("\n".join(summary))
    print(f"\nSaved panel -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        d = os.path.join(HERE, "dataset/yolo_det/images/test")
        img = os.path.join(d, sorted(os.listdir(d))[0]) if os.path.isdir(d) else \
            os.path.join(HERE, "Fire-Segmentation-Dataset-main/images/1000.jpg")
    print(f"Running pipeline on: {img}\n")
    main(img)
