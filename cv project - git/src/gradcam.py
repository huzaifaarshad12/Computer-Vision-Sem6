"""
gradcam.py — Module 4 (Explainability)
Grad-CAM heatmap for the ResNet18 classifier: highlights WHICH pixels pushed
the model toward its decision. Implemented from scratch with forward/backward
hooks on the last conv block (layer4) — no extra dependency needed.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from classifier import FireClassifier, CLASSES


class GradCAM:
    def __init__(self, weights_path, device=None):
        self.clf = FireClassifier(weights_path, device=device)
        self.model = self.clf.model
        self.device = self.clf.device
        self.activations = None
        self.gradients = None
        target = self.model.layer4[-1]  # last residual block
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image, alpha=0.5):
        """image: PIL or path. Returns dict(overlay PIL, label, confidence)."""
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        x = self.clf.transform(image).unsqueeze(0).to(self.device)

        self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].detach()
        idx = int(probs.argmax())
        logits[0, idx].backward()

        # weight activations by mean gradient (importance), ReLU, normalize
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        overlay = self._overlay(image, cam, alpha)
        return {"overlay": overlay, "label": CLASSES[idx], "confidence": float(probs[idx])}

    @staticmethod
    def _overlay(image, cam, alpha):
        w, h = image.size
        cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h))
        cam_np = np.asarray(cam_img, dtype=np.float32) / 255.0
        # simple "jet-like" colormap: blue(low) -> red(high)
        heat = np.zeros((h, w, 3), dtype=np.float32)
        heat[..., 0] = np.clip(1.5 - np.abs(cam_np - 1.0) * 2, 0, 1)  # red
        heat[..., 1] = np.clip(1.5 - np.abs(cam_np - 0.5) * 2, 0, 1)  # green
        heat[..., 2] = np.clip(1.5 - np.abs(cam_np - 0.0) * 2, 0, 1)  # blue
        base = np.asarray(image, dtype=np.float32) / 255.0
        blended = np.clip(base * (1 - alpha) + heat * alpha, 0, 1)
        return Image.fromarray(np.uint8(blended * 255))
