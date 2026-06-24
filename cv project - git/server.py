"""
🔥 Fire & Smoke Vision System — FastAPI backend
================================================
Serves the trained models behind a small JSON API and hosts the React frontend.

Endpoints:
    GET  /                -> the React web app (web/index.html)
    GET  /api/status      -> which models are available
    POST /api/analyze     -> run all modules on an uploaded image, return results

Run:
    python server.py
Then open http://localhost:8000 in your browser.
"""
import base64
import io
import os
import sys

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def _find(*candidates):
    for c in candidates:
        p = os.path.join(HERE, c)
        if os.path.exists(p):
            return p
    return None


CLASSIFIER_W = _find("models/fire_model.pth", "fire_model.pth")
DETECTOR_W = _find("models/fire_detect.pt")
SEGMENTER_W = _find("models/fire_seg.pt")

# lazy model caches
_clf = _cam = _det = _seg = None


def classifier():
    global _clf
    if _clf is None:
        from classifier import FireClassifier
        _clf = FireClassifier(CLASSIFIER_W)
    return _clf


def gradcam():
    global _cam
    if _cam is None:
        from gradcam import GradCAM
        _cam = GradCAM(CLASSIFIER_W)
    return _cam


def detector():
    global _det
    if _det is None:
        from detector import FireDetector
        _det = FireDetector(DETECTOR_W)
    return _det


def segmenter():
    global _seg
    if _seg is None:
        from segmenter import FireSegmenter
        _seg = FireSegmenter(SEGMENTER_W)
    return _seg


def to_data_url(pil_img):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


app = FastAPI(title="Fire & Smoke Vision System")


@app.get("/api/status")
def status():
    return {
        "classifier": bool(CLASSIFIER_W),
        "detector": bool(DETECTOR_W),
        "segmenter": bool(SEGMENTER_W),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  modules: str = Form("classify,detect,segment,explain")):
    """`modules` is a comma-separated list of: classify, detect, segment, explain.
    Only the requested (and available) modules are run."""
    mods = {m.strip() for m in modules.split(",") if m.strip()}
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    result = {"input": to_data_url(image)}

    # Module 1 — classification
    if CLASSIFIER_W and "classify" in mods:
        c = classifier().predict(image)
        result["classify"] = {
            "label": c["label"],
            "confidence": round(c["confidence"] * 100, 1),
            "probs": {k: round(v * 100, 1) for k, v in c["probs"].items()},
        }

    # Module 4 — Grad-CAM (explainability)
    if CLASSIFIER_W and "explain" in mods:
        g = gradcam().generate(image)
        result["gradcam"] = {"image": to_data_url(g["overlay"]),
                             "label": g["label"],
                             "confidence": round(g["confidence"] * 100, 1)}

    # Module 2 — detection
    if DETECTOR_W and "detect" in mods:
        d = detector().predict(image)
        result["detect"] = {
            "image": to_data_url(d["annotated"]),
            "count": d["count"],
            "detections": [
                {"label": x["label"], "confidence": round(x["confidence"] * 100)}
                for x in d["detections"]
            ],
        }

    # Module 3 + 5 — segmentation + severity
    if SEGMENTER_W and "segment" in mods:
        s = segmenter().predict(image)
        result["segment"] = {
            "image": to_data_url(s["annotated"]),
            "count": s["count"],
            "severity": s["severity"],
            "per_class": s["per_class"],
        }

    return JSONResponse(result)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "web", "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("\n  Fire & Smoke Vision System")
    print("  Open  http://localhost:8000  in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
