import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.15),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.15),
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = MNISTCNNModel().to(device)
