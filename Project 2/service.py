from __future__ import annotations

import io
import base64
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pydantic import BaseModel

import bentoml


# ── Pydantic input schemas ────────────────────────────────────────────────────

class ArrayInput(BaseModel):
    # 28×28 pixels as a nested list of ints, e.g. [[0,255,...], ...]
    image: list[list[int]]


class JsonInput(BaseModel):
    # Either flat 784-int list OR base64 PNG string
    pixels: list[int] | None = None
    image_b64: str | None = None


# ── Service ───────────────────────────────────────────────────────────────────

@bentoml.service(
    name="mnist_classifier",
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class MNISTClassifier:

    def __init__(self) -> None:
        self.model = bentoml.picklable_model.load_model("mnist_cnn:latest")
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _run(self, arr: np.ndarray) -> dict:
        """Shared inference logic. arr must be float32 (1,1,28,28)."""
        tensor = torch.tensor(arr, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        predicted  = int(np.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, predicted])
        return {
            "digit": predicted,
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs[0]],
        }

    @bentoml.api()
    def predict_array(self, image: ArrayInput) -> dict:
        """
        Input:  {"image": [[int, ...], ...]}  — 28×28 nested list, values 0-255
        Output: {"digit": int, "confidence": float, "probabilities": list[float]}
        """
        arr = np.array(image.image, dtype=np.float32) / 255.0   # (28,28)
        arr = arr[np.newaxis, np.newaxis, :, :]                   # (1,1,28,28)
        return self._run(arr)

    @bentoml.api()
    def predict_json(self, body: JsonInput) -> dict:
        """
        Input:  {"pixels": [<784 uint8 ints>]}
             or {"image_b64": "<base64-encoded PNG/JPEG>"}
        Output: {"digit": int, "confidence": float, "probabilities": list[float]}
        """
        if body.image_b64:
            img_bytes = base64.b64decode(body.image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
            arr = np.array(img, dtype=np.float32) / 255.0
        elif body.pixels:
            arr = np.array(body.pixels, dtype=np.float32).reshape(28, 28) / 255.0
        else:
            return {"error": "Provide 'image_b64' (base64 PNG) or 'pixels' (784 ints)."}

        arr = arr[np.newaxis, np.newaxis, :, :]  # (1,1,28,28)
        return self._run(arr)