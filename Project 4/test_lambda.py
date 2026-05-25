import base64
import requests

LAMBDA_URL = "..."
IMAGE_PATH = "./example.png"

with open(IMAGE_PATH, "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode("utf-8")

payload = {
    "body": {
        "image_b64": base64_string,
        "top_k": 5
    }
}

print("Sending image to AWS Lambda ResNet-18...")
response = requests.post(LAMBDA_URL, json=payload)

if response.status_code == 200:
    print("\n--- Inference Success ---")
    print(response.json())
else:
    print(f"\n--- Error {response.status_code} ---")
    print(response.text)