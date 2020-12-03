from torch import nn
from .parameterized_cnns import ConvParameterized, OneLayerPreParameterized
from .parameterizing_cnns import ResNetParameterizer, OneLayerResNetPreParameterizer


class ShallowResNetMeta(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.parameterizing = ResNetParameterizer(in_channels=in_channels,
                                                  conv_channels=(32, out_channels),
                                                  conv_ks=(7, 1))
        self.parameterized = ConvParameterized()

    def forward(self, image):
        weights, biases = self.parameterizing(image)
        output = self.parameterized(image, weights, biases)
        return output


class OneLayerPreResNetMeta(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.parameterizing = OneLayerResNetPreParameterizer(in_channels=in_channels)
        self.parameterized = OneLayerPreParameterized(in_channels=in_channels, out_channels=out_channels)

    def forward(self, image):
        weight, bias = self.parameterizing(image)
        output = self.parameterized(image, weight, bias)
        return output
