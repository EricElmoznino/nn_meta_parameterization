from torch import nn
from torchvision.models import resnet18


class OneLayerResNetParameterizer(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, out_k=7):
        super().__init__()

        self.weight_shape = (out_channels, in_channels, out_k, out_k)
        self.weight_size = out_channels * in_channels * out_k * out_k
        self.bias_size = out_channels

        resnet = resnet18()
        resnet.fc = nn.Linear(512, self.weight_size + self.bias_size)

        self.layers = resnet

    def forward(self, image):
        x = self.layers(image)

        weight, bias = x.split([self.weight_size, self.bias_size], dim=1)
        weight = weight.view(weight.size(0), *self.weight_shape)

        return weight, bias
