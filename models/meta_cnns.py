from torch import nn
from .parameterized_cnns import *
from .parameterizing_cnns import *


class OneLayerResNetMeta(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.parameterizing = OneLayerResNetParameterizer(in_channels=in_channels)
        self.parameterized = OneLayerParameterized(in_channels=in_channels, out_channels=out_channels)

    def forward(self, image):
        weight, bias = self.parameterizing(image)
        output = self.parameterized(image, weight, bias)
        return output
