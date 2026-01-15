import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions: int):
        super().__init__()
        c, h, w = input_shape
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.model(x)
