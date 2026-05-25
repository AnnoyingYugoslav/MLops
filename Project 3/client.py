"""
client.py — Test script for the ResNet18 ImageNet classifier API.

Uses ONLY the `requests` library (standard assignment requirement).

Usage:
    python client.py                         # targets localhost:3000
    python client.py --host <EC2-IP> --port 3000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request

import requests


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_image_as_b64(source: str) -> str:
    """
    Load an image from a URL or local file path and return a base64 string.
    """
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=15) as r:
            raw = r.read()
    else:
        with open(source, "rb") as f:
            raw = f.read()
    return base64.b64encode(raw).decode()


def _pretty(resp: requests.Response) -> None:
    print(f"  Status : {resp.status_code}")
    try:
        print(f"  Body   : {json.dumps(resp.json(), indent=4)}\n")
    except Exception:
        print(f"  Body   : {resp.text[:500]}\n")


# ── Demo ───────────────────────────────────────────────────────────────────────

# A few freely-licensed sample images from Wikimedia Commons
SAMPLE_IMAGES = {
    "golden_retriever": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/"
        "320px-Golden_Retriever_Dukedestiny01_drvd.jpg"
    ),
    "tabby_cat": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg"
    ),
    "sports_car": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "1/1b/2019_Ferrari_GTC4Lusso.jpg/320px-2019_Ferrari_GTC4Lusso.jpg"
    ),
}


def demo(host: str, port: int) -> None:
    base_url = f"http://{host}:{port}"

    print(f"\n{'=' * 60}")
    print(f"  ResNet18 ImageNet Classifier  →  {base_url}")
    print(f"{'=' * 60}\n")

    # ── Health check ──────────────────────────────────────────────────────────
    print("── Health check ─────────────────────────────────────────────")
    resp = requests.get(f"{base_url}/healthz", timeout=5)
    _pretty(resp)

    # ── Request 1: predict_image (base64 upload) ──────────────────────────────
    label = "golden_retriever"
    url   = SAMPLE_IMAGES[label]
    print(f"── Request 1: POST /predict_image  ({label}) ──────────────")
    print(f"   Fetching sample image from Wikimedia…")
    b64 = _load_image_as_b64(url)
    resp = requests.post(
        f"{base_url}/predict_image",
        json={"image_b64": b64, "top_k": 5},
        timeout=30,
    )
    _pretty(resp)

    # ── Request 2: predict_url  ───────────────────────────────────────────────
    label = "tabby_cat"
    url   = SAMPLE_IMAGES[label]
    print(f"── Request 2: POST /predict_url  ({label}) ──────────────────")
    resp = requests.post(
        f"{base_url}/predict_url",
        json={"url": url, "top_k": 3},
        timeout=30,
    )
    _pretty(resp)

    # ── Request 3: predict_url (sports car) ───────────────────────────────────
    label = "sports_car"
    url   = SAMPLE_IMAGES[label]
    print(f"── Request 3: POST /predict_url  ({label}) ──────────────────")
    resp = requests.post(
        f"{base_url}/predict_url",
        json={"url": url, "top_k": 3},
        timeout=30,
    )
    _pretty(resp)


CURL_EXAMPLES = r"""
╔══════════════════════════════════════════════════════════════╗
║  cURL Examples                                               ║
╚══════════════════════════════════════════════════════════════╝

# 1. Health check
curl -s http://localhost:3000/healthz

# 2. Classify a local image file (base64 upload)
IMAGE_B64=$(python3 -c "
import base64
with open('my_photo.jpg', 'rb') as f:
    print(base64.b64encode(f.read()).decode())
")
curl -s -X POST http://localhost:3000/predict_image \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"top_k\": 5}" | python3 -m json.tool

# 3. Classify from a public URL
curl -s -X POST http://localhost:3000/predict_url \
  -H "Content-Type: application/json" \
  -d '{
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/320px-Golden_Retriever_Dukedestiny01_drvd.jpg",
        "top_k": 5
      }' | python3 -m json.tool
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test client for the ResNet18 ImageNet classifier."
    )
    parser.add_argument("--host", default="localhost",
                        help="Service host (default: localhost)")
    parser.add_argument("--port", default=3000, type=int,
                        help="Service port (default: 3000)")
    args = parser.parse_args()

    print(CURL_EXAMPLES)
    demo(args.host, args.port)
