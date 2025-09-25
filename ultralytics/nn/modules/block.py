# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, Faster_Block, BiFPN_WConcat, LDConv, CBAM
from .transformer import TransformerBlock, DeformableTransformerDecoderLayer

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fPara",
    "Fusion",
    "C2fAttn",
    "C2fCBam",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "RepVGGDW",
    "RepCross",
    "CIB",
    "C2fCIB",
    "C2fMSC",
    "Attention",
    "PSA",
    "SCDown",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class HGFBlock(nn.Module):
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU(), fusion_type='concat'):
        super().__init__()
        block = LightConv if lightconv else Conv

        # Tạo ModuleList cho n lớp
        self.m = nn.ModuleList([
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        ])

        # Khởi tạo lớp Fusion
        self.fusion = Fusion([c1] + [cm] * n, fusion=fusion_type)

        # Squeeze và Excitation convolution
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)

        # Xác định có dùng shortcut connection không
        self.add = shortcut and c1 == c2

    def forward(self, x):
        outputs = [x]  # Lưu đầu vào ban đầu

        # Duyệt qua từng lớp trong ModuleList
        for m in self.m:
            y = m(outputs[-1])  # Tính đầu ra của lớp hiện tại
            outputs.append(y)  # Lưu đầu ra để hợp nhất sau

        # Dùng lớp Fusion để hợp nhất tất cả các đầu ra
        y = self.fusion(outputs)

        # Qua squeeze và excitation convolution
        y = self.ec(self.sc(y))

        # Trả về y + x nếu có shortcut, ngược lại chỉ trả về y
        return y + x if self.add else y
class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        # y = list(channel_shuffle(self.cv1(x),4).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fPara(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing.
        """
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels

        # Convolution đầu tiên: Giảm chiều đầu vào
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Convolution cuối cùng để hợp nhất tất cả các đặc trưng
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # ModuleList chứa n lớp Bottleneck
        self.m = nn.ModuleList([
            Bottleneck(self.c * (i + 1), self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for i in range(n)
        ])
        # self.m = nn.ModuleList([
        #     CrossConvPro(self.c * (i + 1), self.c, 3, 1, g, 1, shortcut) for i in range(n)
        # ])

    def forward(self, x):
        """Forward pass through C2f layer with dense connections."""

        # Tách đầu vào ban đầu thành 2 phần
        y = list(self.cv1(x).chunk(2, 1))  # [y_0, y_1]

        # Duyệt qua từng Bottleneck, mỗi đầu vào là concat các đầu ra trước đó
        for i, m in enumerate(self.m):
            # Nối tất cả các đầu ra trước đó để làm đầu vào cho Bottleneck hiện tại
            bottleneck_input = torch.cat(y[1:], dim=1)
            bottleneck_input = channel_shuffle(bottleneck_input, 4)

            # Đầu ra của Bottleneck hiện tại
            out = m(bottleneck_input)

            # Thêm đầu ra mới vào danh sách các đầu ra
            y.append(out)

        # Sau khi có tất cả đầu ra, nối lại để qua convolution cuối cùng
        y_concat = torch.cat(y, dim=1)

        # Qua convolution cuối để hợp nhất đặc trưng
        return self.cv2(y_concat)

# class CrossConvPro(nn.Module):
#     # Cross Convolution Downsample
#     def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#
#         # Khởi tạo các Convolutions
#         self.cv1 = Conv(c1, c_, (1, k), (1, s))
#         self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
#         self.cv5 = Conv(c1, c_, (1, 7), (1, s))
#         self.cv6 = Conv(c_, c2, (7, 1), (s, 1), g=g)
#
#         # Khởi tạo lớp Fusion
#         self.fusion3 = Fusion([c2, c2, c2], fusion='bifpn')  # Lựa chọn kiểu fusion 'bifpn'
#         self.fusion2 = Fusion([ c2, c2], fusion='bifpn')  # Lựa chọn kiểu fusion 'bifpn'
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """Performs feature sampling, expanding, and applies shortcut if channels match."""
#         attn_0 = self.cv1(x)
#         attn_0 = self.cv2(attn_0)
#
#         attn_2 = self.cv5(x)
#         attn_2 = self.cv6(attn_2)
#
#
#         # Áp dụng shortcut nếu cần
#         return self.fusion3([x, attn_0, attn_2]) if self.add else self.fusion2([attn_0, attn_2])


# class RepCross(torch.nn.Module):
#     def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False) -> None:
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv5 = Conv(c1, c_, (1, k), (1, s), p=(0, k // 2))
#         self.cv6 = Conv(c_, c2, (k, 1), (s, 1), g=g, p=(k // 2, 0))
#         # self.conv = self.cv2(self.cv1)
#         # self.cv3 = Conv(c1, c_, (1, 5), (1, s),padding=(0, 5//2))
#         # self.cv4 = Conv(c_, c2, (5, 1), (s, 1), g=g,padding=(5//2,0))
#         self.cv1 = Conv(c1, c_, (1, 7), (1, s), p=(0, 7 // 2))
#         self.cv2 = Conv(c_, c2, (7, 1), (s, 1), g=g, p=(7 // 2, 0))
#         # self.conv1 = self.cv5(self.cv6)
#         self.add = shortcut and c1 == c2
#         # self.act = nn.SiLU()
#
#     def forward(self, x):
#         attn_0 = self.cv1(x)
#         attn_2 = self.cv5(x)
#         x = attn_0 + attn_2
#         attn_0 = self.cv2(x)
#         attn_2 = self.cv6(x)
#         # attn_1 = self.cv3(x)
#         # attn_1 = self.cv4(attn_1)
#         attn = attn_0 + attn_2
#         # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#         return x + attn if self.add else attn
#
#     def forward_fuse(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
#     @torch.no_grad()
#     def fuse(self):
#         cv1 = fuse_conv_and_bn(self.cv1.conv, self.cv1.bn)
#         cv5 = fuse_conv_and_bn(self.cv5.conv, self.cv5.bn)
#
#         cv2 = fuse_conv_and_bn(self.cv2.conv, self.cv2.bn)
#         cv6 = fuse_conv_and_bn(self.cv6.conv, self.cv6.bn)
#
#         cv1_w = cv1.weight
#         cv1_b = cv1.bias
#         cv5_w = cv5.weight
#         cv5_b = cv5.bias
#
#         cv2_w = cv2.weight
#         cv2_b = cv2.bias
#         cv6_w = cv6.weight
#         cv6_b = cv6.bias
#
#         # cv5_w = torch.nn.functional.avg_pool2d(cv5_w, (1,3), 2)
#         # cv6_w = torch.nn.functional.avg_pool2d(cv6_w, (3,1), 2)
#
#         print(cv5_w.shape)
#         print(cv6_w.shape)
#         cv5_w = torch.nn.functional.pad(cv5_w, [2, 2, 0, 0])
#         cv6_w = torch.nn.functional.pad(cv6_w, [0, 0, 2, 2])
#
#
#
#         final_conv_w = cv1_w + cv5_w
#         final_conv_b = cv1_b + cv5_b
#         final1_conv_w = cv2_w + cv6_w
#         final1_conv_b = cv2_b + cv6_b
#
#         print(cv1_w.shape)
#         print(final_conv_w.shape)
#         print(cv5_w.shape)
#
#         print(cv2_w.shape)
#         print(final1_conv_w.shape)
#         print(cv6_w.shape)
#
#         cv1.weight.data.copy_(final_conv_w)
#         cv1.bias.data.copy_(final_conv_b)
#         cv2.weight.data.copy_(final1_conv_w)
#         cv2.bias.data.copy_(final1_conv_b)
#
#         self.cv1 = cv1
#         self.cv2 = cv2
#         del self.cv5
#         del self.cv6
class CrossConvDilated(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = Conv(c1, c_, (1, k), (1, s),padding=(0, k//2))
        # self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g,padding=(k//2,0))
        # # self.cv3 = Conv(c1, c_, (1, 5), (1, s),padding=(0, 5//2))
        # # self.cv4 = Conv(c_, c2, (5, 1), (s, 1), g=g,padding=(5//2,0))
        # self.cv5 = Conv(c1, c_, (1, 7), (1, s),padding=(0, 7//2))
        # self.cv6 = Conv(c_, c2, (7, 1), (s, 1), g=g,padding=(7//2,0))
        self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv5_1 = Conv(c1, c_, (1, k), (1, s), d=5)
        self.cv5_2 = Conv(c_, c2, (k, 1), (s, 1), d=5)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        x1 = self.cv1_1(x)
        x1 = self.cv1_2(x1)

        x2 = self.cv2_1(x)
        x2 = self.cv2_2(x2)

        x5 = self.cv5_1(x)
        x5 = self.cv5_2(x5)

        output = torch.cat((x1, x2, x5), dim=1)
        output = channel_shuffle(output, 4)
        output = self.squeeze(output)

        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + output if self.add else output

class RepCross(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False) -> None:
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv5 = Conv(c1, c_, (1, k), (1, s))
        self.cv6 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        # self.conv = self.cv2(self.cv1)
        # self.cv3 = Conv(c1, c_, (1, 5), (1, s),padding=(0, 5//2))
        # self.cv4 = Conv(c_, c2, (5, 1), (s, 1), g=g,padding=(5//2,0))
        self.cv1 = Conv(c1, c_, (1, 7), (1, s))
        self.cv2 = Conv(c_, c2, (7, 1), (s, 1), g=g)
        # self.conv1 = self.cv5(self.cv6)
        self.add = shortcut and c1 == c2
        # self.act = nn.SiLU()

    def forward(self, x):
        attn_0 = self.cv1(x)
        attn_2 = self.cv5(x)
        x0 = x + attn_0 + attn_2
        attn_0 = self.cv2(x0)
        attn_2 = self.cv6(x0)
        # attn_1 = self.cv3(x)
        # attn_1 = self.cv4(attn_1)
        attn = x0 + attn_0 + attn_2
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + attn if self.add else attn
class CrossConvPro(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        # self.cv3 = Conv(c1, c_, (1, 5), (1, s),padding=(0, 5//2))
        # self.cv4 = Conv(c_, c2, (5, 1), (s, 1), g=g,padding=(5//2,0))
        self.cv5 = Conv(c1, c_, (1, 7), (1, s))
        self.cv6 = Conv(c_, c2, (7, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        attn_0 = self.cv1(x)
        attn_0 = self.cv2(attn_0)

        # attn_1 = self.cv3(x)
        # attn_1 = self.cv4(attn_1)

        attn_2 = self.cv5(x)
        attn_2 = self.cv6(attn_2)
        attn = attn_0 + attn_2
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + attn if self.add else attn

class C2fMSC(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList((RepCross(self.c, self.c, 3, 1, g, 1, shortcut) for _ in range(n)))
        # self.concat = BiFPN_WConcat(inc_list=[self.c] * (2 + n), dimension=1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        # y = list(channel_shuffle(self.cv1(x), 4).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        # return self.cv2(self.concat(y))
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        # return self.cv2(self.concat(y))

# ################### MHA begin #############################
# class MHA(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MHA, self).__init__()
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#         # Khởi tạo các lớp tuyến tính cho Q, K, V
#         self.q_linear = nn.Linear(embed_dim, embed_dim)
#         self.k_linear = nn.Linear(embed_dim, embed_dim)
#         self.v_linear = nn.Linear(embed_dim, embed_dim)
#         self.fc_out = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, x):
#         if x.dim() == 4:  # Đảm bảo rằng x có đúng số chiều
#             t, batch_size, seq_length, embed_dim = x.size()
#         else:
#             raise ValueError(f"Expected input with 3 dimensions, but got {x.size()}")
#
#         # batch_size, seq_length, embed_dim = x.size()
#
#
#         # Tạo Q, K, V từ x
#         Q = self.q_linear(x)
#         K = self.k_linear(x)
#         V = self.v_linear(x)
#
#         # Chia thành các đầu attention
#         Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#
#         # Tính attention scores bằng scaled dot-product
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attention = torch.softmax(scores, dim=-1)
#
#         # Tính giá trị attention
#         out = torch.matmul(attention, V)
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
#         out = self.fc_out(out)
#
#         return out
#
# ################### MHA end #############################
class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))
        self.m = nn.Sequential(*(CrossConvPro(self.c_, self.c_, 3, 1, g, 1, shortcut) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
        # self.m = DeformableTransformerDecoderLayer(c_, 8)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""
    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(RepCross(self.c, self.c, 3, 1, g, 1, shortcut) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)
        # self.attn = MHA(self.c, 3)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))
class C2fCBam(nn.Module):
    """C2f module with an additional attn module."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.m = nn.ModuleList(RepCross(self.c, self.c, 3, 1, g, 1, shortcut) for _ in range(n))
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = CBAM(self.cv2,  7)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)

#
# class ADown(nn.Module):
#     """ADown."""
#
#     def __init__(self, c1, c2):
#         """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
#
#     def forward(self, x):
#         """Forward pass through ADown layer."""
#         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x1, x2 = x.chunk(2, 1)
#         x1 = self.cv1(x1)
#         x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
#         x2 = self.cv2(x2)
#         return torch.cat((x1, x2), 1)


# class AConv(nn.Module):
#     """ADown."""
#
#     def __init__(self, c1, c2):
#         """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, c1, 3, 2, 1)
#         self.cvdw = Conv(c1, c2, 7, 2, 3, g=c1)
#         # self.cv1 = LDConv(c1 // 2, self.c, 5, 2, 1)
#         self.cv2 = Conv(c2, 2 * c1, 1, 1, 0)
#         self.cv3 = Conv(2 * c1, c2, 1, 1, 0)
#         self.cv4 = Conv(c1, c1, 1, 1, 0)
#
#     def forward(self, x):
#         """Forward pass through ADown layer."""
#         # x1 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         # x = torch.nn.functional.max_pool2d(x, 3, 2, 1)
#         # x1, x2 = x.chunk(2, 1)
#         x1 = self.cv1(x)
#         x2 = torch.nn.functional.max_pool2d(x, 3, 2, 1)
#         # x2 = self.cv4(x2)
#         x3 = self.cv2(self.cvdw(x))
#         # x3 = self.cvdw(self.cv2(x))
#         x0 = torch.cat((x1, x2), 1)
#         return self.cv3(x0 + x3)

class ADown(nn.Module):
    """ADown."""
    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, c1, 3, 2, 1)
        # self.cbam = CBAM(c1)
        self.cvdw = Conv(c1, c2, 7, 2, 3, g=c1)
        # self.cvdw = RepVGGDW(c1,c2)
        # self.cv1 = LDConv(c1 // 2, self.c, 5, 2, 1)
        self.cv2 = Conv(c2, c1, 1, 1, 0)
        self.cv3 = Conv(c1, c2, 1, 1, 0)
        # self.cv4 = Conv(c1, c1, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        # x1 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        # x = torch.nn.functional.max_pool2d(x, 3, 2, 1)
        # x1, x2 = x.chunk(2, 1)
        # x = self.cbam(x)
        x1 = self.cv1(x)
        x2 = torch.nn.functional.max_pool2d(x, 3, 2, 1)
        # x2 = self.cv4(x2)
        x3 = self.cv2(self.cvdw(x)) #best
        # x3 = self.cvdw(x) #repvggdw
        # x0= torch.cat((x1, x2), 1)
        return self.cv3(x1+x2+x3)
class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed, od=None) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        # self.conv = Conv(ed, od, 7, 2, 3, g=ed, act=False)
        # self.conv1 = Conv(ed, od, 3, 2, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


# class CIB(nn.Module):
#     """
#     Conditional Identity Block (CIB) module.
#
#     Args:
#         c1 (int): Number of input channels.
#         c2 (int): Number of output channels.
#         shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
#         e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
#         lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
#     """
#
#     def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
#         """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = nn.Sequential(
#             Conv(c1, c1, 3, g=c1),
#             Conv(c1, 2 * c_, 1),
#             RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
#             Conv(2 * c_, c2, 1),
#             Conv(c2, c2, 3, g=c2),
#         )
#
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """
#         Forward pass of the CIB module.
#
#         Args:
#             x (torch.Tensor): Input tensor.
#
#         Returns:
#             (torch.Tensor): Output tensor.
#         """
#         return x + self.cv1(x) if self.add else self.cv1(x)
#
#
# class C2fCIB(C2f):
#     """
#     C2fCIB class represents a convolutional block with C2f and CIB modules.
#
#     Args:
#         c1 (int): Number of input channels.
#         c2 (int): Number of output channels.
#         n (int, optional): Number of CIB modules to stack. Defaults to 1.
#         shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
#         lk (bool, optional): Whether to use local key connection. Defaults to False.
#         g (int, optional): Number of groups for grouped convolution. Defaults to 1.
#         e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
#     """
#
#     def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
#         """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):
    """
    Position-wise Spatial Attention module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention module, and feed-forward network with channel reduction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Forward pass of the PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))
# class SCDown(nn.Module):
#     """Spatial Channel Downsample (SCDown) module for reducing spatial and channel dimensions."""
#
#     def __init__(self, c1, c2, k, s):
#         """
#         Spatial Channel Downsample (SCDown) module.
#
#         Args:
#             c1 (int): Number of input channels.
#             c2 (int): Number of output channels.
#             k (int): Kernel size for the convolutional layer.
#             s (int): Stride for the convolutional layer.
#         """
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
#
#     def forward(self, x):
#         """
#         Forward pass of the SCDown module.
#
#         Args:
#             x (torch.Tensor): Input tensor.
#
#         Returns:
#             (torch.Tensor): Output tensor after applying the SCDown module.
#         """
#         return self.cv2(self.cv1(x))
def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out



class IncepCrossConvPro(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * 0.5)  # hidden channels
        self.cv0 = Conv(c1, c_, 1, 1)
        self.cv1 = Conv(c_, c_, (1, k), (1, s),p=(0, k//2))
        self.cv2 = Conv(c_, c_, (k, 1), (s, 1), g=g,p=(k//2,0))
        # self.cv1 = Conv(c_, c_, (1, 5), (1, s),padding=(0, 5//2))
        # self.cv2 = Conv(c_, c_, (5, 1), (s, 1), g=g,padding=(5//2,0))
        self.cv5 = Conv(c_, c_, (1, 7), (1, s),p=(0, 7//2))
        self.cv6 = Conv(c_, c_, (7, 1), (s, 1), g=g,p=(7//2,0))
        self.ch_cv = Conv(c1, c2, 1, 1)
        # self.ca = CoordAtt(c1, c1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        x1 = self.cv0(x)
        attn_0 = self.cv1(x1)
        attn_0 = self.cv2(attn_0)

        # attn_1 = self.cv3(x)
        # attn_1 = self.cv4(attn_1)
        x2 = self.cv0(x)
        attn_2 = self.cv5(x2)
        attn_2 = self.cv6(attn_2)
        x3 = torch.cat((attn_0, attn_2), 1)
        x3 = channel_shuffle(x3, 4)
        # x3 = self.ch_cv(x3)
        # x3 = self.ca(channel_shuffle(x3, 4))
        return x + x3 if self.add else x3

class C3IMSC(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(IncepCrossConvPro(c_, c_, 3, 1, g, 0.5, shortcut) for _ in range(n)))