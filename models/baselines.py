import torch
from torch import nn


class ShallowCNN(nn.Module):

    def __init__(self, in_channels=3, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, image):
        output = self.layers(image)
        return output
