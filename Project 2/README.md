# MNIST Model Serving — Homework 2

## Chosen Framework: **BentoML**

BentoML was chosen for its first-class PyTorch support, dead-simple API definition,
built-in batching, and the ability to package everything (model + code + dependencies)
into a single deployable `Bento` image with one command.

---

## Repository Structure

```
.
├── train.py              # Original training script (Homework 1)
├── save_model.py         # Exports Lightning checkpoint → BentoML model store
├── service.py            # BentoML service (two API endpoints)
├── client.py             # Python client demo + cURL examples
├── bentofile.yaml        # BentoML build manifest
├── Dockerfile            # Original training image
├── Dockerfile.serve      # Lightweight inference image
├── docker-compose.yml    # Updated: adds `bentoml` service (port 3000)
├── requirements.txt      # Training dependencies
└── requirements.serve.txt# Serving dependencies (CPU torch)
```

---

## Step-by-Step Guide

### 1 · Train the model (Homework 1 recap)

```bash
# Start MLflow + trainer
docker compose up -d mlflow trainer

# Run training inside the trainer container
docker exec -it model_trainer python train.py
# Checkpoint is saved to ./checkpoints/best-model.ckpt
```

### 2 · Start the BentoML serving API

```bash
docker compose up -d bentoml
```

The `bentoml` container automatically:
1. Calls `save_model.py` to load `checkpoints/best-model.ckpt` and register
   it in the BentoML model store as **`mnist_cnn:latest`**.
2. Launches `bentoml serve service:svc` on **port 3000**.

Check it's healthy:
```bash
curl http://localhost:3000/healthz
# → {"status": "ok"}
```

Swagger UI is available at **http://localhost:3000**.

---

### 3 · Send inference requests

#### Option A — Python `requests`

```bash
python client.py          # runs all three demo requests
python client.py --host localhost --port 3000
```

**Example output:**
```
── Request 2: POST /predict/json (flat pixel list) ──────────
  Status : 200
  Body   : {
      "digit": 7,
      "confidence": 0.9983,
      "probabilities": [
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9983, 0.0, 0.0002
      ]
  }
```

#### Option B — `cURL` with a flat pixel list

```bash
# Generate 784 zeros as a placeholder (all-black image → digit 0 most likely)
PIXELS=$(python3 -c "print(','.join(['0']*784))")

curl -s -X POST http://localhost:3000/predict/json \
  -H "Content-Type: application/json" \
  -d "{\"pixels\": [$PIXELS]}" | python3 -m json.tool
```

#### Option C — `cURL` with a base64-encoded PNG

```bash
# Convert any 28×28 grayscale PNG to base64 and POST it
IMAGE_B64=$(python3 -c "
import base64
with open('my_digit.png', 'rb') as f:
    print(base64.b64encode(f.read()).decode())
")

curl -s -X POST http://localhost:3000/predict/json \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\"}" | python3 -m json.tool
```

---

## API Reference

### `POST /predict/array`

| Field       | Value                                      |
|-------------|-------------------------------------------|
| Content-Type| `application/json`                         |
| Body        | JSON-serialised 2-D array `[[0,255,...],…]`|
| Response    | `{"digit": int, "confidence": float, "probabilities": [float×10]}` |

### `POST /predict/json`

| Field       | Value |
|-------------|-------|
| Content-Type| `application/json` |
| Body (option 1) | `{"pixels": [<784 uint8 ints>]}` |
| Body (option 2) | `{"image_b64": "<base64-encoded PNG/JPEG>"}` |
| Response    | `{"digit": int, "confidence": float, "probabilities": [float×10]}` |

---

## Packaging as a Standalone Bento (optional)

```bash
# Inside the serving container or any env with bentoml installed:
bentoml build

# The resulting Bento can be containerised:
bentoml containerize mnist_classifier:latest

# And pushed to a registry:
docker push your-registry/mnist_classifier:latest
```

---

## Notes

* The serving container uses **CPU-only PyTorch** to keep the image small (~2 GB).
  For GPU inference, replace the torch wheel in `requirements.serve.txt` with the
  appropriate CUDA variant and add the `deploy.resources` GPU block to the
  `bentoml` service in `docker-compose.yml`.
* BentoML's built-in **adaptive batching** groups concurrent requests automatically;
  no code changes are needed.
* The model store is persisted in the named Docker volume `bentoml_store` so the
  checkpoint export step is skipped on subsequent container restarts (unless the
  volume is removed).
