import requests
import base64

# Load and encode the image
with open("example.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# Send request to the server
resp = requests.post(
    "http://13.60.173.14:3000/predict_image",
    json={"body": {"image_b64": b64, "top_k": 5}}
)

print(resp.json())