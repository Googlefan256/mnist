from model import model, device
from data import test_loader
import torch

correct = 0
total = 0
model.load_state_dict(torch.load("state.pt", weights_only=True))
model.eval()
with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print(f"Accuracy: {100 * correct / total}%")
