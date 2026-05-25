import time
import io
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from train import MNISTModel 

def quantize_int16(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                w = module.weight.data
                max_val = torch.max(torch.abs(w))
                if max_val > 0:
                    scale = 32767.0 / max_val
                    w_q = torch.clamp(torch.round(w * scale), -32768, 32767)
                    module.weight.data = w_q / scale
    return model

def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return size_mb

def benchmark_inference(checkpoint_path, dataloader, precision="fp32"):
    device = torch.device("cpu")
    
    model = MNISTModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    
    if precision == "int8":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif precision == "int16":
        model = quantize_int16(model)
    elif precision == "fp16":
        model = model.half()

    model_size_mb = get_model_size_mb(model)

    all_preds = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            if precision == "fp16":
                images = images.half()

            if precision == "int16":
                start_time = time.perf_counter()
                features = model.layers[:7](images).cpu().numpy().flatten().astype(np.int16)
                weight_matrix = model.layers[7].weight.data.cpu().numpy().astype(np.int16)
                _ = np.dot(weight_matrix, features)
                
                outputs = model(images)
                end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                outputs = model(images)
                end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_latency = np.mean(latencies)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return f1, avg_latency, model_size_mb

def main():
    checkpoint_path = "checkpoints/best-model.ckpt" 
    
    test_data = MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    precisions = ["fp32", "fp16", "int8", "int16"]
    
    print("--- Benchmark Results ---")
    print(f"{'Precision':<10} | {'Model Size':<12} | {'F1 Score':<15} | {'Time (ms)':<10}")
    print("-" * 68)
    
    for precision in precisions:
        f1, latency, model_size = benchmark_inference(checkpoint_path, test_loader, precision)
        print(f"{precision:<10} | {model_size:<9.2f} MB | {f1:<15.4f} | {latency:.4f} ms")

if __name__ == "__main__":
    main()