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


class Pro_align(nn.Module):
    def __init__(self, args):
        super(Pro_align, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.conv1x1 = nn.Conv2d(self.nf * 3, self.nf, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv1_3x3 = nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, l1, l2, l3):
        r1 = self.lrelu(self.conv3x3(l1))
        r2 = self.lrelu(self.conv3x3(l2))
        r3 = self.lrelu(self.conv3x3(l3))
        fuse = self.lrelu(cat_conv(self.conv1x1, [r1, r2, r3]))
        r1 = self.lrelu(cat_conv(self.conv1_3x3, [r1, fuse]))
        r2 = self.lrelu(cat_conv(self.conv1_3x3, [r2, fuse]))
        r3 = self.lrelu(cat_conv(self.conv1_3x3, [r3, fuse]))
        return l1 + r1, l2 + r2, l3 + r3


class SR(nn.Module):
    def __init__(self, args):
        super(SR, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)
        self.Pro_align = Pro_align(args)
        self.conv1x1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def upsample(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return self.lrelu(self.conv1x1(x))

    def forward(self, l1, l2, l3):
        l1, l2, l3 = self.Pro_align(l1, l2, l3)
        return tuple(self.upsample(i) for i in (l1, l2, l3))


class DR(nn.Module):
    def __init__(self, args):
        super(DR, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (1 / self.args.upscale_factor, 1 / self.args.upscale_factor)
        self.Pro_align = Pro_align(args)
        self.conv = nn.Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def downsample(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return self.lrelu(self.conv(x))

    def forward(self, l1, l2, l3):
        l1 = self.downsample(l1)
        l2 = self.downsample(l2)
        l3 = self.downsample(l3)
        return self.Pro_align(l1, l2, l3)


class Up_projection(nn.Module):
    def __init__(self, args):
        super(Up_projection, self).__init__()
        self.args = args
        self.SR = SR(args)
        self.DR = DR(args)
        self.SR1 = SR(args)

    def forward(self, l1, l2, l3):
        h1, h2, h3 = self.SR(l1, l2, l3)
        d1, d2, d3 = self.DR(h1, h2, h3)
        r1, r2, r3 = d1 - l1, d2 - l2, d3 - l3
        s1, s2, s3 = self.SR1(r1, r2, r3)
        return h1 + s1, h2 + s3, h3 + s3


class Down_projection(nn.Module):
    def __init__(self, args):
        super(Down_projection, self).__init__()
        self.args = args
        self.SR = SR(args)
        self.DR = DR(args)
        self.DR1 = DR(args)

    def forward(self, h1, h2, h3):
        l1, l2, l3 = self.DR(h1, h2, h3)
        s1, s2, s3 = self.SR(l1, l2, l3)
        r1, r2, r3 = s1 - h1, s2 - h2, s3 - h3
        d1, d2, d3 = self.DR1(r1, r2, r3)
        return l1 + d1, l2 + d2, l3 + d3