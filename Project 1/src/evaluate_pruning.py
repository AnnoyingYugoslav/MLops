import time
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.utils.prune as prune

from train import MNISTModel 

def count_nonzero_params(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)

def apply_unstructured_pruning(model, amount=0.5):
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model

def apply_structured_pruning_and_shrink(model, amount=0.5):
    linear_layer = model.layers[7] 
    next_layer = model.layers[9]   
    
    weight = linear_layer.weight.data
    bias = linear_layer.bias.data
    
    neuron_norms = torch.norm(weight, p=1, dim=1)
    
    num_to_keep = int(weight.size(0) * (1 - amount))
    _, top_indices = torch.topk(neuron_norms, k=num_to_keep)
    top_indices = torch.sort(top_indices).values 
    
    new_linear = torch.nn.Linear(weight.size(1), num_to_keep)
    new_linear.weight.data = weight[top_indices, :]
    new_linear.bias.data = bias[top_indices]
    
    new_next = torch.nn.Linear(num_to_keep, next_layer.weight.data.size(0))
    new_next.weight.data = next_layer.weight.data[:, top_indices]
    new_next.bias.data = next_layer.bias.data
    
    model.layers[7] = new_linear
    model.layers[9] = new_next
    
    return model

def benchmark_inference(checkpoint_path, dataloader, pruning_variant="baseline", amount=0.5):
    device = torch.device("cpu")
    model = MNISTModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    if pruning_variant == "unstructured":
        model = apply_unstructured_pruning(model, amount=amount)
    elif pruning_variant == "structured":
        model = apply_structured_pruning_and_shrink(model, amount=amount)

    non_zero_params = count_nonzero_params(model)
    all_preds = []
    all_labels = []
    latencies = []

    # Benchmark Loop
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_latency = np.mean(latencies)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return f1, avg_latency, non_zero_params

def main():
    checkpoint_path = "checkpoints/best-model.ckpt" 
    
    test_data = MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    variants = ["baseline", "unstructured", "structured"]
    pruning_amount = 0.50
    
    print(f"\n--- Pruning Analysis (Pruning Amount: {pruning_amount*100}%) ---")
    print(f"{'Variant':<25} | {'F1 Score':<10} | {'Inference Time':<15} | {'Non-zero Params'}")
    print("-" * 72)
    
    for variant in variants:
        f1, latency, nz_params = benchmark_inference(checkpoint_path, test_loader, variant, amount=pruning_amount)
        variant_name = "Baseline (no pruning)" if variant == "baseline" else f"{variant.capitalize()} pruning"
        print(f"{variant_name:<25} | {f1:<10.4f} | {latency:.4f} ms      | {nz_params}")

if __name__ == "__main__":
    main()