from torch import nn
from torch.nn import functional as F
from .custom_layers import batchconv2d


class ConvParameterized(nn.Module):

    def forward(self, x, weights, biases):
        for i, (w, b) in enumerate(zip(weights, biases)):
            if w.size(3) > 1 or w.size(4) > 1:
                x = F.pad(x, pad=[w.size(4) // 2, w.size(4) // 2,
                                  w.size(3) // 2, w.size(3) // 2], mode='replicate')

            x = batchconv2d(x, w, b)

            if i < len(weights) - 1:
                x = F.relu(x, inplace=True)

        return x


class OneLayerPreParameterized(nn.Module):

    def __init__(self, in_channels=3, out_channels=None, hidden_channels=32):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        self.hidden_channels = hidden_channels

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, image, weight, bias):
        image = F.pad(image, pad=[weight.size(4) // 2, weight.size(4) // 2,
                                  weight.size(3) // 2, weight.size(3) // 2], mode='replicate')
        x = batchconv2d(image, weight, bias)
        x = self.relu(x)

        x = self.conv2(x)

        return x
