import argparse
import torch
import bentoml
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import MNISTModel


def save(checkpoint_path: str) -> None:

    print(f"Loading checkpoint: {checkpoint_path}")
    lit_model = MNISTModel.load_from_checkpoint(checkpoint_path)
    lit_model.eval()

    class MNISTInferenceModel(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    inference_model = MNISTInferenceModel(lit_model.layers)

    saved = bentoml.picklable_model.save_model(
        "mnist_cnn",
        inference_model,
        metadata={
            "description": "CNN trained on MNIST via PyTorch Lightning + Optuna",
            "input": "float32 tensor (N, 1, 28, 28), values in [0, 1]",
            "output": "float32 logits (N, 10)",
        },
    )
    print(f"Model saved to BentoML store: {saved.tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best-model.ckpt")
    args = parser.parse_args()
    save(args.checkpoint)