"""
segmenter.py — Module 3 inference (YOLOv8 segmentation) + Module 5 (severity)
Produces a pixel-level mask of fire/smoke and measures how much of the
image is covered, which feeds the severity score.
"""
import numpy as np
from PIL import Image

import severity


class FireSegmenter:
    def __init__(self, weights_path, conf=0.25):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(weights_path)
        self.conf = conf

    def predict(self, image_path):
        """Returns dict(annotated, fire_fraction, severity, count, per_class)."""
        results = self.model.predict(image_path, conf=self.conf, verbose=False)
        r = results[0]
        annotated = Image.fromarray(r.plot()[:, :, ::-1])  # BGR -> RGB

        h, w = r.orig_shape
        total_px = float(h * w)
        union = np.zeros((h, w), dtype=bool)
        per_class = {}

        if r.masks is not None:
            names = r.names
            classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
            for m, cls in zip(r.masks.data.cpu().numpy(), classes):
                mask = m.astype(bool)
                # masks may be at model resolution; resize bool mask to image size
                if mask.shape != (h, w):
                    mask = np.array(Image.fromarray(mask).resize((w, h))).astype(bool)
                union |= mask
                name = names[int(cls)]
                per_class[name] = per_class.get(name, 0) + int(mask.sum())

        fire_fraction = float(union.sum()) / total_px if total_px else 0.0
        return {
            "annotated": annotated,
            "fire_fraction": fire_fraction,
            "severity": severity.score(fire_fraction),
            "count": 0 if r.masks is None else len(r.masks.data),
            "per_class": {k: round(v / total_px * 100, 2) for k, v in per_class.items()},
        }
