from typing import List, Tuple, Optional, Union
from torch import Tensor, Size
from torch.nn import functional as F

_size = Union[Size, List[int], Tuple[int, ...]]


def batchconv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, _size] = 1,
                padding: Union[int, _size] = 0, dilation: Union[int, _size] = 1) -> Tensor:
    assert input.ndim == 4
    assert weight.ndim == 5
    assert weight.size(0) == input.size(0)
    assert weight.size(2) == input.size(1)
    if bias is not None:
        assert bias.ndim == 2
        assert bias.size(0) == input.size(0)
        assert bias.size(1) == weight.size(1)

    bs, out_c, in_c, kh, kw = weight.size()

    input = input.view(1, bs * in_c, input.size(2), input.size(3))
    weight = weight.contiguous().view(bs * out_c, in_c, kh, kw)
    if bias is not None:
        bias = bias.contiguous().view(bs * out_c)

    output = F.conv2d(input, weight, bias, groups=bs, stride=stride, padding=padding, dilation=dilation)
    output = output.view(bs, out_c, output.size(2), output.size(3))

    return output
