"""
service.py — BentoML service wrapping a pretrained ResNet18 ImageNet classifier.

Endpoints
---------
POST /predict_image   — accepts a base64-encoded JPEG/PNG and returns top-5 predictions
POST /predict_url     — accepts a public image URL and returns top-5 predictions
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

import bentoml

# ── ImageNet class labels ──────────────────────────────────────────────────────
# Loaded once at import time from the bundled JSON file.
_LABELS_PATH = Path(__file__).parent / "imagenet_classes.json"
with open(_LABELS_PATH) as f:
    IMAGENET_CLASSES: list[str] = json.load(f)   # list of 1000 class names

# ── Pre-processing pipeline (standard ImageNet) ────────────────────────────────
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Pydantic input schemas ─────────────────────────────────────────────────────

class ImageInput(BaseModel):
    """Base64-encoded image (JPEG, PNG, …)."""
    image_b64: str
    top_k: int = 5


class UrlInput(BaseModel):
    """Publicly accessible image URL."""
    url: str
    top_k: int = 5


# ── Helper ─────────────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a (1, 3, 224, 224) float32 tensor."""
    img = img.convert("RGB")
    return _PREPROCESS(img).unsqueeze(0)   # (1, 3, 224, 224)


def _top_k_result(logits: torch.Tensor, top_k: int) -> dict:
    probs = F.softmax(logits, dim=-1)[0]   # (1000,)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, 1000))
    predictions = [
        {
            "rank": i + 1,
            "class_id": int(idx),
            "label": IMAGENET_CLASSES[int(idx)],
            "confidence": round(float(p), 4),
        }
        for i, (idx, p) in enumerate(zip(top_indices, top_probs))
    ]
    return {
        "top_prediction": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "top_k": predictions,
    }


# ── Service ────────────────────────────────────────────────────────────────────

@bentoml.service(
    name="resnet18_classifier",
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class ImageNetClassifier:

    def __init__(self) -> None:
        self.model = bentoml.picklable_model.load_model("resnet18_imagenet:latest")
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _infer(self, tensor: torch.Tensor, top_k: int) -> dict:
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return _top_k_result(logits.cpu(), top_k)

    @bentoml.api()
    def predict_image(self, body: ImageInput) -> dict:
        """
        Input:  {"image_b64": "<base64-encoded image>", "top_k": 5}
        Output: {"top_prediction": str, "confidence": float, "top_k": [...]}
        """
        img_bytes = base64.b64decode(body.image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        tensor = _pil_to_tensor(img)
        return self._infer(tensor, body.top_k)

    @bentoml.api()
    def predict_url(self, body: UrlInput) -> dict:
        """
        Input:  {"url": "https://…/cat.jpg", "top_k": 5}
        Output: {"top_prediction": str, "confidence": float, "top_k": [...]}
        """
        with urllib.request.urlopen(body.url, timeout=10) as resp:
            img_bytes = resp.read()
        img = Image.open(io.BytesIO(img_bytes))
        tensor = _pil_to_tensor(img)
        return self._infer(tensor, body.top_k)
