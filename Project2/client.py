"""
client.py — Example client for the BentoML MNIST serving API.

Usage:
    python client.py [--host localhost] [--port 3000]
"""

from __future__ import annotations

import argparse
import base64
import io
import json

import numpy as np
import requests
from PIL import Image


def _get_sample_digit(digit: int = 7) -> np.ndarray:
    from torchvision.datasets import MNIST
    from torchvision import transforms

    dataset = MNIST(
        root="/tmp/mnist_demo",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    for img, label in dataset:
        if label == digit:
            return (img.squeeze().numpy() * 255).astype(np.uint8)

    raise ValueError(f"Digit {digit} not found")


def _img_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pretty(resp: requests.Response) -> None:
    print(f"  Status : {resp.status_code}")

    try:
        print(f"  Body   : {json.dumps(resp.json(), indent=4)}\n")
    except Exception:
        print(f"  Body   : {resp.text[:500]}\n")


def demo(host: str, port: int) -> None:
    base_url = f"http://{host}:{port}"

    print(f"\n{'=' * 60}")
    print(f"  MNIST BentoML Client Demo  →  {base_url}")
    print(f"{'=' * 60}\n")
    DIGIT = 1
    print(f"Fetching sample MNIST test image (digit = {DIGIT}) …")
    img_arr = _get_sample_digit(digit=DIGIT)

    print(f"  Image shape: {img_arr.shape}, dtype: {img_arr.dtype}\n")

    # ------------------------------------------------------------------
    # Request 1 — /predict_array
    #
    # Service signature:
    #   def predict_array(self, image: ArrayInput)
    #
    # Therefore BentoML expects:
    #   {
    #       "image": {
    #           "image": [[...], [...]]
    #       }
    #   }
    # ------------------------------------------------------------------
    print("── Request 1: POST /predict_array ────────────────────────────")

    resp = requests.post(
        f"{base_url}/predict_array",
        json={
            "image": {
                "image": img_arr.tolist()
            }
        },
    )

    _pretty(resp)

    # ------------------------------------------------------------------
    # Request 2 — /predict_json (flat pixel list)
    #
    # Service signature:
    #   def predict_json(self, body: JsonInput)
    #
    # Therefore BentoML expects:
    #   {
    #       "body": {
    #           "pixels": [...]
    #       }
    #   }
    # ------------------------------------------------------------------
    print("── Request 2: POST /predict_json (flat pixel list) ──────────")

    resp = requests.post(
        f"{base_url}/predict_json",
        json={
            "body": {
                "pixels": img_arr.flatten().tolist()
            }
        },
    )

    _pretty(resp)

    # ------------------------------------------------------------------
    # Request 3 — /predict_json (base64 PNG)
    # ------------------------------------------------------------------
    print("── Request 3: POST /predict_json (base64 PNG) ───────────────")

    resp = requests.post(
        f"{base_url}/predict_json",
        json={
            "body": {
                "image_b64": _img_to_b64(img_arr)
            }
        },
    )

    _pretty(resp)


CURL_EXAMPLES = r"""
╔══════════════════════════════════════════════════════════════╗
║  cURL Examples                                               ║
╚══════════════════════════════════════════════════════════════╝

# 1. Nested pixel array
curl -s -X POST http://localhost:3000/predict_array \
  -H "Content-Type: application/json" \
  -d '{
        "image": {
          "image": [[0,0,...,0],[0,0,...,0],...]
        }
      }' | python3 -m json.tool

# 2. Flat pixel list
curl -s -X POST http://localhost:3000/predict_json \
  -H "Content-Type: application/json" \
  -d '{
        "body": {
          "pixels": [0,0,...,0]
        }
      }' | python3 -m json.tool

# 3. Base64-encoded PNG
IMAGE_B64=$(python3 -c "
import base64
with open('digit.png', 'rb') as f:
    print(base64.b64encode(f.read()).decode())
")

curl -s -X POST http://localhost:3000/predict_json \
  -H "Content-Type: application/json" \
  -d "{\"body\":{\"image_b64\":\"$IMAGE_B64\"}}" \
  | python3 -m json.tool

# 4. Health check
curl -s http://localhost:3000/livez
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=3000, type=int)

    args = parser.parse_args()

    print(CURL_EXAMPLES)

    demo(args.host, args.port)