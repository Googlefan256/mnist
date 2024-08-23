from model import model, device
from data import train_loader
from torch import nn
from torch import optim
import torch
from typing import Dict


def uncompile(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod.") :]  # Remove the prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

model = torch.compile(model, fullgraph=True, dynamic=True, mode="max-autotune")
criterion = nn.NLLLoss()
epoch = 50
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
epoch_len = len(train_loader)
scheduler = get_cosine_schedule_with_warmup(optimizer, epoch_len, epoch_len * epoch - 1)

for epoch in range(epoch):
    total_loss = 0
    for images, labels in tqdm(train_loader):
        with torch.autocast("cuda", torch.bfloat16):
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Loss: {total_loss / epoch_len}")

torch.save(uncompile(model.state_dict()), "state.pt")
