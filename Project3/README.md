# ImageNet Classifier — Homework 3

## Chosen Model: **ResNet18 (pretrained, ImageNet-1K)**

ResNet18 from `torchvision` was selected as a simple, well-understood baseline.
No custom training is needed — the pretrained weights ship with torchvision and
are baked into the Docker image at build time.

## Serving Framework: **BentoML**

BentoML wraps the model in two REST endpoints, handles serialisation, and
provides a built-in health check at `/healthz`.

---

## Repository Structure

```
.
├── save_model.py         # Exports ResNet18 weights → BentoML model store
├── fetch_labels.py       # Downloads ImageNet class names at build time
├── service.py            # BentoML service (two API endpoints)
├── client.py             # Python test client (uses only `requests`)
├── bentofile.yaml        # BentoML build manifest
├── Dockerfile.serve      # CPU-only inference image
├── docker-compose.yml    # Local dev / smoke-test stack
└── requirements.serve.txt
```

---

## Step 1 — Build and test locally

```bash
# Build the serving image (downloads weights + labels inside the image)
docker compose up --build -d

# Smoke test
curl http://localhost:3000/healthz
```

Swagger UI is available at **http://localhost:3000**.

---

## Step 2 — Deploy to AWS EC2

### 2a · Launch an EC2 instance

| Setting        | Value                         |
|----------------|-------------------------------|
| AMI            | Ubuntu 24.04 LTS              |
| Instance type  | `t3.medium` (2 vCPU, 4 GB)   |
| Storage        | 20 GB gp3                     |
| Security group | inbound TCP **3000** + SSH 22 |

### 2b · Install Docker on the instance

```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 2c · Copy files and start the service

```bash
# From your local machine
scp -i your-key.pem -r ./ ubuntu@<EC2-PUBLIC-IP>:~/hw3/

# On the EC2 instance
cd ~/hw3
docker compose up --build -d
```

### 2d · Verify it's running

```bash
curl http://<EC2-PUBLIC-IP>:3000/healthz
# → {"status":"ok"}
```

---

## Step 3 — Run the test client

```bash
# Point the client at your EC2 instance
python client.py --host <EC2-PUBLIC-IP> --port 3000
```

**Example output:**

```
── Request 1: POST /predict_image  (golden_retriever) ──────────────
  Status : 200
  Body   : {
      "top_prediction": "golden retriever",
      "confidence": 0.9241,
      "top_k": [
          {"rank": 1, "class_id": 207, "label": "golden retriever", "confidence": 0.9241},
          {"rank": 2, "class_id": 208, "label": "Labrador retriever", "confidence": 0.0412},
          ...
      ]
  }
```

---

## API Reference

### `POST /predict_image`

Classify an image uploaded as a base64 string.

| Field       | Value                                                         |
|-------------|---------------------------------------------------------------|
| Content-Type| `application/json`                                            |
| Body        | `{"image_b64": "<base64 PNG/JPEG>", "top_k": 5}`             |
| Response    | `{"top_prediction": str, "confidence": float, "top_k": [...]}` |

### `POST /predict_url`

Classify an image fetched from a public URL (server-side download).

| Field       | Value                                                         |
|-------------|---------------------------------------------------------------|
| Content-Type| `application/json`                                            |
| Body        | `{"url": "https://…/image.jpg", "top_k": 5}`                 |
| Response    | `{"top_prediction": str, "confidence": float, "top_k": [...]}` |

### `GET /healthz`

Returns `{"status": "ok"}` when the service is ready.

---

## Optional — Package as a standalone Bento

```bash
# Inside any env with bentoml installed:
bentoml build

# Containerise the Bento:
bentoml containerize resnet18_classifier:latest

# Push to a registry:
docker push your-registry/resnet18_classifier:latest
```

---

## Notes

- The serving container uses **CPU-only PyTorch** (`torch+cpu` wheel) to keep
  the image under ~2 GB. For GPU inference on a `g4dn` instance, swap the torch
  wheel in `requirements.serve.txt` for the matching CUDA variant.
- Model weights and ImageNet labels are downloaded **at image build time**, so
  the container starts instantly with no runtime network calls.
- BentoML's adaptive batching is enabled via the `signatures` block in
  `save_model.py` — concurrent requests are automatically grouped.
