"""
fetch_labels.py — Downloads ImageNet-1K class names and writes imagenet_classes.json.
Called automatically during Docker image build.
"""

import json
import urllib.request

# torchvision bundles the official ILSVRC class index as a JSON file
URL = (
    "https://raw.githubusercontent.com/pytorch/vision/main/"
    "torchvision/models/_meta.py"
)

# Simpler: use the well-known community JSON (same order as torchvision)
LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)

def main():
    print(f"Fetching ImageNet labels from:\n  {LABELS_URL}")
    with urllib.request.urlopen(LABELS_URL, timeout=30) as r:
        labels: list[str] = json.loads(r.read())
    assert len(labels) == 1000, f"Expected 1000 labels, got {len(labels)}"
    with open("imagenet_classes.json", "w") as f:
        json.dump(labels, f)
    print(f"Saved {len(labels)} labels → imagenet_classes.json")


if __name__ == "__main__":
    main()
