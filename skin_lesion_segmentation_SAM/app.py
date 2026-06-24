"""Local Gradio demo: Skin Lesion Segmentation with SAM (zero-shot).

Run:  python app.py
Then open the http://127.0.0.1:7860 link it prints (it also opens your browser).
Upload a skin photo from File Explorer -> click Segment.

First run downloads the SAM ViT-B checkpoint (~375 MB) into this folder.
Runs on CPU (a few seconds per image) if no CUDA GPU is available.
"""
import os
import urllib.request

import numpy as np
import cv2
import torch
import gradio as gr
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------------------------
# 1. Model setup
# ---------------------------------------------------------------------------
SAM_TYPE = "vit_b"
CKPT_PATH = "sam_vit_b_01ec64.pth"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On CPU, make PyTorch use every core (small but free speedup for the encoder).
if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"CPU mode: using {torch.get_num_threads()} threads.")

# Measured on the N=18 hand-annotated test subset (notebook Section 6.2).
METRICS = {"IoU": 0.40, "Dice": 0.54, "Precision": 0.43, "Recall": 0.83, "PixelAcc": 0.59}


def _download_checkpoint():
    if os.path.exists(CKPT_PATH):
        return
    print(f"Downloading SAM checkpoint ({SAM_TYPE}, ~375 MB) -> {CKPT_PATH} ...")

    def _hook(blocks, block_size, total):
        done = blocks * block_size
        pct = min(100, done * 100 // total) if total > 0 else 0
        print(f"\r  {pct:3d}%  ({done // (1024*1024)} MB)", end="")

    urllib.request.urlretrieve(CKPT_URL, CKPT_PATH, _hook)
    print("\nDownload complete.")


_download_checkpoint()
print(f"Loading SAM ({SAM_TYPE}) on {DEVICE} ...")
sam = sam_model_registry[SAM_TYPE](checkpoint=CKPT_PATH).to(DEVICE)
predictor_vitb = SamPredictor(sam)
print("SAM loaded.")

# --- Optional fast model: MobileSAM (ViT-T). Loaded lazily on first "fast mode" use. ---
MOBILE_CKPT = "mobile_sam.pt"
MOBILE_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
_mobile_predictor = None


def get_mobile_predictor():
    """Lazily download + load MobileSAM the first time fast mode is used."""
    global _mobile_predictor
    if _mobile_predictor is None:
        from mobile_sam import sam_model_registry as m_registry, SamPredictor as MSamPredictor
        if not os.path.exists(MOBILE_CKPT):
            print(f"Downloading MobileSAM checkpoint (~40 MB) -> {MOBILE_CKPT} ...")
            urllib.request.urlretrieve(MOBILE_URL, MOBILE_CKPT)
            print("MobileSAM download complete.")
        print("Loading MobileSAM (vit_t) ...")
        m = m_registry["vit_t"](checkpoint=MOBILE_CKPT).to(DEVICE)
        m.eval()
        _mobile_predictor = MSamPredictor(m)
        print("MobileSAM loaded.")
    return _mobile_predictor


# --- Trained model: U-Net fine-tuned on SkinDisNet. Preferred when its checkpoint
#     is present (drop unet_skindisnet.pt from the Colab run into this folder). ---
UNET_CKPT = "unet_skindisnet.pt"
_IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_unet = None


def get_unet():
    """Load the trained U-Net once. Returns None if the checkpoint is absent."""
    global _unet
    if _unet is None:
        if not os.path.exists(UNET_CKPT):
            return None
        import segmentation_models_pytorch as smp
        net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        net.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE))
        net.to(DEVICE).eval()
        _unet = net
        print(f"Loaded trained U-Net: {UNET_CKPT}")
    return _unet


def segment_lesion_unet(img):
    """Trained-U-Net lesion mask (HxW bool) at the image's native resolution."""
    net = get_unet()
    h, w = img.shape[:2]
    x = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    x = (x - _IMNET_MEAN) / _IMNET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEVICE)
    with torch.no_grad():
        pr = torch.sigmoid(net(x))[0, 0].cpu().numpy()
    pr = cv2.resize(pr, (w, h))
    return clean_mask(pr > 0.5)


# ---------------------------------------------------------------------------
# 2. Segmentation (same logic as the notebook)
# ---------------------------------------------------------------------------
def clean_mask(m):
    """Keep the largest connected component and fill holes."""
    from scipy import ndimage
    lbl, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    biggest = lbl == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(biggest)


def segment_lesion(img, predictor):
    """Center-prompted SAM lesion mask (HxW bool); smallest valid region, cleaned."""
    predictor.set_image(img)
    h, w = img.shape[:2]
    # foreground: 3x3 grid in the central part of the frame (the lesion)
    xs = np.linspace(0.35, 0.65, 3) * w
    ys = np.linspace(0.35, 0.65, 3) * h
    fg = [[x, y] for y in ys for x in xs]
    # background: near the four corners -> "this is healthy skin, not lesion"
    bg = [[0.04 * w, 0.04 * h], [0.96 * w, 0.04 * h], [0.04 * w, 0.96 * h], [0.96 * w, 0.96 * h]]
    pts = np.array(fg + bg, dtype=np.float32)
    lbls = np.array([1] * len(fg) + [0] * len(bg))
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=lbls, multimask_output=True)
    cand = [m for m in masks if 0.05 < m.mean() < 0.85]
    best = min(cand, key=lambda m: m.mean()) if cand else masks[int(np.argmax(scores))]
    return clean_mask(best.astype(bool))


def overlay(img, mask, color=(255, 0, 0), alpha=0.45):
    out = img.copy()
    col = np.zeros_like(img)
    col[mask] = color
    out = cv2.addWeighted(out, 1, col, alpha, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 2)
    return out


# ---------------------------------------------------------------------------
# 3. Gradio app
# ---------------------------------------------------------------------------
_HAS_UNET = os.path.exists(UNET_CKPT)
_ACTIVE = "Trained U-Net (ResNet-34), fine-tuned on SkinDisNet" if _HAS_UNET \
    else f"SAM ({SAM_TYPE}), zero-shot"

MODEL_CARD = f"""
## Model card

**Active model:** **{_ACTIVE}**. Running on **{DEVICE.upper()}**.

**Trained model:** a **U-Net (ResNet-34 encoder)** trained on SkinDisNet using SAM's
lesion-focused auto-labels (no human annotation). When `{UNET_CKPT}` is in this folder it
is used for segmentation — one fast forward pass, focused on the lesion rather than the
whole hand. {"It is loaded and active." if _HAS_UNET else "Not found yet, so the app falls back to zero-shot SAM below."}

**Zero-shot fallback (SAM):** the image is prompted with a 3x3 grid of foreground points
in the centre plus four background points at the corners; SAM's most conservative valid
mask is kept, then cleaned (largest component + hole fill).

**Evaluation metrics** (mean over 18 hand-annotated images):

| Metric | Score |
|---|---|
| IoU (Jaccard) | {METRICS['IoU']:.2f} |
| Dice (F1) | {METRICS['Dice']:.2f} |
| Precision | {METRICS['Precision']:.2f} |
| Recall | {METRICS['Recall']:.2f} |
| Pixel Accuracy | {METRICS['PixelAcc']:.2f} |

**Why SAM?** SkinDisNet was published for *classification* and ships **no segmentation
masks**, so supervised training (e.g. U-Net) would require annotating thousands of images
first. SAM is a powerful pretrained, promptable, zero-shot segmenter, giving high-quality
lesion masks immediately; we only hand-annotated a small subset to *measure* accuracy.

**Fast mode:** the ⚡ toggle swaps in **MobileSAM** (ViT‑T, distilled SAM) for ~1–3 s
inference on CPU instead of ~30 s. The metrics above are measured on **ViT‑B** — fast mode
is for a quicker live preview, not the evaluated configuration.
"""


def app_segment(image, fast_mode):
    if image is None:
        return None, None, "Upload a photo first."
    img = np.asarray(image)
    if img.ndim == 2:                       # grayscale -> 3-channel
        img = np.stack([img] * 3, axis=-1)
    img = np.ascontiguousarray(img[..., :3], dtype=np.uint8)   # drop alpha if present
    h, w = img.shape[:2]                    # cap very large phone photos for speed
    scale = 1024 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    unet = None if fast_mode else get_unet()
    if unet is not None:                        # trained model takes priority
        mask = segment_lesion_unet(img)
        model_name = "Trained U-Net (SkinDisNet)"
    else:
        if fast_mode:
            predictor, model_name = get_mobile_predictor(), "MobileSAM (ViT-T, fast)"
        else:
            predictor, model_name = predictor_vitb, f"SAM ({SAM_TYPE}, zero-shot)"
        mask = segment_lesion(img, predictor)
    ov = overlay(img, mask)
    cover = float(mask.mean()) * 100
    return ov, (mask * 255).astype(np.uint8), f"{model_name}: lesion covers {cover:.1f}% of the image."


with gr.Blocks(title="Skin Lesion Segmentation (SAM)") as demo:
    gr.Markdown("# Skin Lesion Segmentation with SAM (zero-shot)")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Upload a photo", sources=["upload"])
        out_ov = gr.Image(type="numpy", label="Segmentation overlay")
        out_mask = gr.Image(type="numpy", label="Binary lesion mask")
    fast = gr.Checkbox(value=False, label="⚡ Fast mode (MobileSAM, ~1-3s) — ViT-B is the evaluated model")
    info = gr.Textbox(label="Result", interactive=False)
    gr.Button("Segment", variant="primary").click(
        app_segment, inputs=[inp, fast], outputs=[out_ov, out_mask, info])
    gr.Markdown(MODEL_CARD)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
