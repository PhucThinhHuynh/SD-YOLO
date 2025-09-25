# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DDWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "LDConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "SEAttention",
    "CoordAtt",
    "ECAAttention",
    "Concat",
    "BiFPN_Concat3",
    "BiFPN_Concat2",
    "BiFPN_WConcat3",
    "BiFPN_WConcat2",
    "RepConv",
    "Faster_Block",
    "BiFPN_WConcat"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

from einops import rearrange
class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  # nn.BatchNorm2d(outc),
                                  # nn.SiLU()
                                  )  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32, k=None):  # inp: number of input channel, oup: number of output channel
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        return out
from torch.nn import init
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, k=None, x=None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (ECA-Net).
    - Tanpa MLP/reduction: pakai GAP (B,C,1,1) -> Conv1d(1,1,k) di dimensi channel -> sigmoid -> scale.
    Args:
        c (int): jumlah channel input
        gamma (int|float): faktor skala untuk menghitung k
        b (int|float): bias untuk menghitung k
        k_size (int|None): kalau None, dihitung adaptif dari c; kalau diberikan, harus bilangan ganjil >= 3
    """
    def __init__(self, c: int, gamma: float = 2, b: float = 1, k_size: int | None = None):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(c) + b) / gamma))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)  # minimal 3 dan selalu ganjil
        else:
            if k_size % 2 == 0 or k_size < 3:
                raise ValueError("k_size must be odd and >= 3")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        y = y.unsqueeze(1)                             # [B, 1, C]
        y = self.conv(y)                               # [B, 1, C]
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y
from timm.models.layers import DropPath
# class Partial_conv3(nn.Module):
#     def __init__(self, dim, n_div, forward):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x):
#         # only for inference
#         x = x.clone()  # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
#         return x
#
#     def forward_split_cat(self, x):
#         # for training/inference
#         # x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1, x2 = torch.split(x, [self.partial_conv3.weight.size(1), x.size(1) - self.partial_conv3.weight.size(1)],
#                              dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)
#         # x = channel_shuffle(x, 4)
#         # Can use channel shuffle
#         return x
# class Faster_Block(nn.Module):
#     def __init__(self,
#                  inc,   #input
#                  dim,   # output
#                  n_div=4, #divide partical conv
#                  mlp_ratio=2,
#                  drop_path=0.1, #regularization
#                  layer_scale_init_value=0.0,
#                  pconv_fw_type='split_cat'
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.n_div = n_div
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         mlp_layer = [
#             Conv(dim, mlp_hidden_dim, 1),
#             nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
#         ] # block after PConv
#
#         self.mlp = nn.Sequential(*mlp_layer)
#
#         self.spatial_mixing = Partial_conv3(
#             dim,
#             n_div,
#             pconv_fw_type
#         ) #PConv or call spatial_mixing
#
#         self.adjust_channel = None
#         if inc != dim:
#             self.adjust_channel = Conv(inc, dim, 1)
#
#         if layer_scale_init_value > 0:
#             self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.forward = self.forward_layer_scale
#         else:
#             self.forward = self.forward
#
#     def forward(self, x):
#         if self.adjust_channel is not None: #only for input != output
#             x = self.adjust_channel(x)
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(self.mlp(x))
#         return x
#
#     def forward_layer_scale(self, x):
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(
#             self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
#         return x

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 2, 1, bias=False)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1, x2 = torch.split(x, [self.partial_conv3.weight.size(1), x.size(1) - self.partial_conv3.weight.size(1)],
                             dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.mp(x2)
        x = torch.cat((x1, x2), 1)
        # x = channel_shuffle(x, 4)
        # Can use channel shuffle
        return x
class Faster_Block(nn.Module):
    def __init__(self,
                 inc,   #input
                 dim,   # output
                 n_div=4, #divide partical conv
                 mlp_ratio=2,
                 drop_path=0.1, #regularization
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ] # block after PConv

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        ) #PConv or call spatial_mixing
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None: #only for input != output
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.mp(shortcut) + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.mp(shortcut) + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# class DDWConv(nn.Module):
#     def __init__(self, c1, c2, k=3, s=2, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv1 = DWConv(c1, c2, k, s, act=act)
#         # self.mp = torch.nn.functional.max_pool2d(x, 3, 2, 1)
#         self.conv2 = Conv(c2, c2, k=1, s=1)
#
#     def forward(self, x):
#         """Apply 2 convolutions to input tensor."""
#         # x1 = torch.nn.functional.max_pool2d(x, 3, 2, 1)
#         return self.conv2(self.conv1(x))
class DDWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, g=False, d=0, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        if g==True:
            g = c1
        else:
            g = 8
        self.conv1 = Conv(c1, c2, k, s, g=g, d=d, act=act)
        self.kz = k
        self.conv2 = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x3 = torch.cat((x1, x2),1)

        return self.conv2(self.conv1(x))
class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

######################################################################## MSCBAM
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) #thu gọn h và w
        # self.conv = nn.Conv2d(channels, channels, 1, 1, autopad(1, 1, 1), groups=1, dilation=1, bias=False)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True,)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))

class RepCross(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False) -> None:
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv5 = nn.Conv2d(c1, c2, (1, 3), (1, s), padding=(0, 3//2))
        # self.cv6 = nn.Conv2d(c2, c2, (3, 1), (s, 1),padding=(3//2,0))
        # self.cv7 = nn.Conv2d(c1, c2, (1, 9), (1, s), padding=(0, 9 // 2))
        # self.cv8 = nn.Conv2d(c2, c2, (9, 1), (s, 1), padding=(9 // 2, 0))
        # self.cv1 = nn.Conv2d(c1, c2, (1, 7), (1, s), padding=(0, 7//2))
        # self.cv2 = nn.Conv2d(c2, c2, (7, 1), (s, 1),padding=(7//2,0))
        self.cv5 = nn.Conv2d(c1, c2, (1, 3), (1, s), padding=(0, 3 // 2))
        self.cv6 = nn.Conv2d(c2, c2, (3, 1), (s, 1), padding=(3 // 2, 0))
        self.cv7 = nn.Conv2d(c1, c2, (1, 9), (1, s), padding=(0, 9 // 2))
        self.cv8 = nn.Conv2d(c2, c2, (9, 1), (s, 1), padding=(9 // 2, 0))
        self.cv1 = nn.Conv2d(c1, c2, (1, 7), (1, s), padding=(0, 7 // 2))
        self.cv2 = nn.Conv2d(c2, c2, (7, 1), (s, 1), padding=(7 // 2, 0))
        # self.conv1 = self.cv5(self.cv6)
        self.add = shortcut and c1 == c2
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_0 = self.cv1(x)
        attn_2 = self.cv5(x)
        attn_3 = self.cv7(x)
        # x0 = x + attn_0 + attn_2
        # attn_0 = self.cv2(attn_0)
        # attn_2 = self.cv6(attn_2)
        # attn_3 = self.cv8(attn_3)
        attn_0 = self.cv2(attn_0)
        attn_2 = self.cv6(attn_2)
        attn_3 = self.cv8(attn_3)
        # attn_1 = self.cv3(x)
        # attn_1 = self.cv4(attn_1)
        attn = self.act(attn_0) + self.act(attn_2) + self.act(attn_3)
        # attn = attn_0 + attn_2 + attn_3
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return attn
class SpatialAttention(nn.Module): ####new SA
    """Spatial-attention module."""

    def __init__(self, kernel_size=7, channels=None):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.cv1 = RepCross(2, 1, kernel_size)
        # self.cv1 = LDConv(2, 1, 3, bias=False)
        # self.act = nn.Sigmoid()
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        # return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
        return x * self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))

# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""
#
#     def __init__(self, kernel_size=7, channels=None):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
#         padding = 3 if kernel_size == 7 else 1
#         # self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.cv1 = RepCross(2, 1, kernel_size)
#         # self.cv1 = LDConv(2, 1, 3, bias=False)
#         self.act = nn.Sigmoid()
#     def forward(self, x):
#         """Apply channel and spatial attention on input for feature recalibration."""
#         # return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
#         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=3, k=None, x=None):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(3, c1)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        # return self.spatial_attention(self.channel_attention(x))
        return self.channel_attention(self.spatial_attention(x))
######################################################
# class ChannelAttention(nn.Module):
#     """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
#
#     def __init__(self, channels: int) -> None:
#         """Initializes the class and sets the basic configurations and instance variables required."""
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
#         return x * self.act(self.fc(self.pool(x)))
#
#
# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""
#
#     def __init__(self, kernel_size=7):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
#         padding = 3 if kernel_size == 7 else 1
#         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         """Apply channel and spatial attention on input for feature recalibration."""
#         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
#
#
# class CBAM(nn.Module):
#     """Convolutional Block Attention Module."""
#
#     def __init__(self, c1, kernel_size=7, k=None, x=None):
#         """Initialize CBAM with given input channel (c1) and kernel size."""
#         super().__init__()
#         self.channel_attention = ChannelAttention(c1)
#         self.spatial_attention = SpatialAttention(7)
#
#     def forward(self, x):
#         """Applies the forward pass through C1 module."""
#         return self.spatial_attention(self.channel_attention(x))

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

class BiFPN_WConcat(nn.Module):
    def __init__(self, inc_list, dimension=1):
        super(BiFPN_WConcat, self).__init__()
        self.d = dimension
        # self.relu = nn.SiLU()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
    def forward(self, x):
        # w = self.relu(self.w.clone())
        w = self.w.clone()
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion [fusion_weight[i] * x[i] for i in range(len(x))]
        x = [weight[i] * x[i] for i in range(len(x))]
        # return torch.cat(x, self.d)
        return channel_shuffle(torch.cat(x, self.d),4)

class BiFPN_WConcat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_WConcat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.relu(self.w.clone())
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        # return torch.cat(x, self.d)
        return channel_shuffle(torch.cat(x, self.d), 4)
class BiFPN_WConcat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_WConcat3, self).__init__()
        self.d = dimension
        self.relu = nn.ReLU()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.relu(self.w.clone())
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        # return torch.cat(x, self.d)
        return channel_shuffle(torch.cat(x, self.d),4)
def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


