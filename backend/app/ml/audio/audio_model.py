import torch
import torch.nn as nn


class AudioCNN(nn.Module):

    def __init__(self, embedding_size: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding_layer = nn.Linear(
            128,
            embedding_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)

        # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)

        # [batch, 256]
        embedding = self.embedding_layer(x)

        return embedding