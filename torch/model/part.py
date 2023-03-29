from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import torchvision.ops

from .util import cat_conv


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        self.init_weights(self.conv1)
        self.init_weights(self.conv2)

    @staticmethod
    def init_weights(conv):
        init.kaiming_normal_(conv.weight, a=0, mode='fan_in')
        conv.weight.data *= 0.1  # for residual block
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class DCN_sep(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_channels_features: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 bias: bool = True,
                 mask: bool = True):
        super(DCN_sep, self).__init__()

        self.dcn = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias)

        kernel_size_ = _pair(kernel_size)
        offset_channels = deformable_groups * kernel_size_[0] * kernel_size_[1]

        self.conv_offset = nn.Conv2d(in_channels_features, offset_channels * 2, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, bias=True)
        self.conv_mask = nn.Conv2d(in_channels_features, offset_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=True) if mask else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor, feature: torch.Tensor):
        offset = self.conv_offset(feature)
        mask = torch.sigmoid(self.conv_mask(feature)) if self.conv_mask else None

        return self.dcn(input, offset, mask)


class PCDLayer(nn.Module):
    """ Alignment module using Pyramid, Cascading and Deformable convolution"""
    def __init__(self, args, first_layer: bool):
        super(PCDLayer, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups

        self.offset_conv1 = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.dcnpack = DCN_sep(self.nf, self.nf, self.nf, 3, stride=1, padding=1, dilation=1,
                               deformable_groups=self.groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not first_layer:
            self.offset_conv2 = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)
            self.fea_conv = nn.Conv2d(2 * self.nf, self.nf, 3, 1, 1)

    def forward(self, current_sources: Tuple[torch.Tensor, torch.Tensor],
                last_offset: torch.Tensor, last_feature: torch.Tensor):
        offset = self.lrelu(cat_conv(self.offset_conv1, current_sources))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = offset.shape
            last_offset = last_offset[..., :h, :w]
            offset = self.lrelu(cat_conv(self.offset_conv2, (offset, last_offset * 2)))
        offset = self.lrelu(self.offset_conv3(offset))
        feature = self.dcnpack(current_sources[0], offset)
        if last_feature is not None:
            last_feature = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = feature.shape
            last_feature = last_feature[..., :h, :w]
            feature = cat_conv(self.fea_conv, (feature, last_feature))
        feature = self.lrelu(feature)
        return offset, feature
