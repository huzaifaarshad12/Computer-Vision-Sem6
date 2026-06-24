"""
detector.py — Module 2 inference (YOLOv8 detection)
Draws bounding boxes showing WHERE fire / smoke is in the image.
"""
from PIL import Image


class FireDetector:
    def __init__(self, weights_path, conf=0.25):
        from ultralytics import YOLO  # lazy import so the app starts without ultralytics
        self.model = YOLO(weights_path)
        self.conf = conf

    def predict(self, image_path):
        """Returns dict(annotated: PIL.Image, detections: list, count: int)."""
        results = self.model.predict(image_path, conf=self.conf, verbose=False)
        r = results[0]
        # r.plot() returns a BGR numpy array with boxes drawn
        annotated = Image.fromarray(r.plot()[:, :, ::-1])  # BGR -> RGB

        detections = []
        names = r.names
        if r.boxes is not None:
            for b in r.boxes:
                cls = int(b.cls[0])
                detections.append({
                    "label": names[cls],
                    "confidence": float(b.conf[0]),
                    "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                })
        return {"annotated": annotated, "detections": detections, "count": len(detections)}
