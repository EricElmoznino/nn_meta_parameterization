from torch import nn
from torchvision.models import resnet18


class ResNetParameterizer(nn.Module):

    def __init__(self, in_channels, conv_channels, conv_ks):
        super().__init__()

        self.weight_shapes, self.weight_sizes, self.bias_sizes, self.param_sizes = [], [], [], []
        in_c = in_channels
        for out_c, k in zip(conv_channels, conv_ks):
            self.weight_shapes.append((out_c, in_c, k, k))
            self.weight_sizes.append(out_c * in_c * k * k)
            self.bias_sizes.append(out_c)
            self.param_sizes.append(self.weight_sizes[-1] + self.bias_sizes[-1])
            in_c = out_c

        resnet = resnet18()
        resnet.fc = nn.Linear(512, sum(self.param_sizes))

        self.layers = resnet

    def forward(self, image):
        x = self.layers(image)

        params = x.split(self.param_sizes, dim=1)
        params = [p.split([ws, bs], dim=1) for p, ws, bs in zip(params, self.weight_sizes, self.bias_sizes)]
        weights, biases = zip(*params)
        weights = tuple(w.view(w.size(0), *wshape) for w, wshape in zip(weights, self.weight_sizes))

        return weights, biases


class OneLayerResNetPreParameterizer(nn.Module):

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
