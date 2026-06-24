"""
classifier.py — Module 1 inference
Loads the trained ResNet18 and classifies an image as fire_smoke / normal.
Returns the label AND a confidence score (softmax probability).
Transforms here MUST match the training transforms (ImageNet normalization).
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image

CLASSES = ["fire_smoke", "normal"]   # ImageFolder alphabetical order
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FireClassifier:
    def __init__(self, weights_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASSES))
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, image):
        """image: PIL.Image or path. Returns dict(label, confidence, probs)."""
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0].cpu()
        idx = int(probs.argmax())
        return {
            "label": CLASSES[idx],
            "confidence": float(probs[idx]),
            "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        }
