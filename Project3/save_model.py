"""
save_model.py — Downloads pretrained ResNet18 and saves it to the BentoML model store.
Run once before starting the serving container.

Usage:
    python save_model.py
"""

import torch
import bentoml
from torchvision.models import resnet18, ResNet18_Weights


def save() -> None:
    print("Loading pretrained ResNet18 (ImageNet weights)…")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval()

    saved = bentoml.picklable_model.save_model(
        "resnet18_imagenet",
        model,
        metadata={
            "description": "torchvision ResNet18 pretrained on ImageNet-1K",
            "input": "float32 tensor (N, 3, 224, 224), ImageNet-normalised",
            "output": "float32 logits (N, 1000)",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
        },
        signatures={
            "__call__": {"batchable": True, "batch_dim": 0},
        },
    )
    print(f"Saved to BentoML store: {saved.tag}")


if __name__ == "__main__":
    save()
