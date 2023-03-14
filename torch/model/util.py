import functools
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_use_fold_catconv = False


def use_fold_catconv(value=True):
    global _use_fold_catconv
    _use_fold_catconv = value


def cat_simp(ts, *args, **kwargs):
    """auto eliminate cat if there's only one input"""
    if len(ts) == 1:
        return ts[0]
    return torch.cat(ts, *args, **kwargs)


def cat_conv(conv: nn.Conv2d, tensors, scale=None):
    """separate cat+conv into multiple conv to reduce memory footprint"""
    if _use_fold_catconv:
        w = conv.weight.detach()
        b = conv.bias.detach()
        if scale is not None:
            w *= scale
            b *= scale
        output = None
        channels = [0]
        channels.extend(itertools.accumulate(int(tensor.shape[1]) for tensor in tensors))
        for ti, cb, ce in zip(tensors, channels, channels[1:]):
            c = ti.shape[1]
            convi = nn.Conv2d(c, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                              conv.dilation, bias=output is None).eval()
            convi.weight = nn.Parameter(w[:, cb:ce, :, :], requires_grad=False)
            if output is None:
                convi.bias = nn.Parameter(b, requires_grad=False)
            outputi = convi(ti)
            output = outputi if output is None else output + outputi
        return output
    else:
        return conv(torch.cat(tensors, dim=1))


# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
class normalizer:
    nm = torchvision.transforms.Normalize
    sqrt2 = math.sqrt(2)

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), kr=0.2126, kb=0.0722, depth=8):
        self.mean = mean
        self.std = std
        self.krgb = (kr, 1 - kr - kb, kb)
        self.depth = depth
        self.uv_bias = (1 << (depth - 1)) / ((1 << depth) - 1)

    @staticmethod
    def _inv(mean, std):
        inv_std = tuple(1 / i for i in std)
        inv_mean = tuple(-j * i for i, j in zip(inv_std, mean))
        return inv_mean, inv_std

    def _yuv_dist(self):
        rm, gm, bm = self.mean
        rs, gs, bs = self.std
        kr, kg, kb = self.krgb

        ym = rm * kr + gm * kg + bm * kb
        ys = math.sqrt((rs * kr) ** 2 + (gs * kg) ** 2 + (bs * kb) ** 2)
        um = (bm - ym) / (1 - kb) / 2 + self.uv_bias
        us = math.sqrt(bs * bs + ys * ys) / (1 - kb) / 2
        vm = (rm - ym) / (1 - kr) / 2 + self.uv_bias
        vs = math.sqrt(rs * rs + ys * ys) / (1 - kr) / 2
        return [ym, um, vm], [ys, us, vs]

    def normalize_rgb(self, rgb: torch.Tensor):
        return self.nm(self.mean, self.std)(rgb)

    def denormalize_rgb(self, rgb: torch.Tensor):
        return self.nm(*self._inv(self.mean, self.std))(rgb)

    def normalize_yuv_444(self, yuv: torch.Tensor):
        return self.nm(*self._yuv_dist())(yuv)

    def denormalize_yuv_444(self, yuv: torch.Tensor):
        return self.nm(*self._inv(*self._yuv_dist()))(yuv)

    def _normalize_yuv_42x(self, y: torch.Tensor, uv: torch.Tensor, scale):
        mean, std = self._yuv_dist()
        std[1], std[2] = std[1] * scale, std[2] * scale
        y = self.nm(mean[0], std[0])(y)
        uv = self.nm(mean[1:], std[1:])(uv)
        return y, uv

    def _denormalize_yuv_42x(self, y: torch.Tensor, uv: torch.Tensor, scale):
        mean, std = self._yuv_dist()
        std[1], std[2] = std[1] * scale, std[2] * scale
        mean, std = self._inv(mean, std)
        y = self.nm(mean[0], std[0])(y)
        uv = self.nm(mean[1:], std[1:])(uv)
        return y, uv

    def normalize_yuv_422(self, y: torch.Tensor, uv: torch.Tensor):
        return self._normalize_yuv_42x(y, uv, 1 / self.sqrt2)

    def denormalize_yuv_422(self, y: torch.Tensor, uv: torch.Tensor):
        return self._denormalize_yuv_42x(y, uv, 1 / self.sqrt2)

    def normalize_yuv_420(self, y: torch.Tensor, uv: torch.Tensor):
        return self._normalize_yuv_42x(y, uv, 1 / 2)

    def denormalize_yuv_420(self, y: torch.Tensor, uv: torch.Tensor):
        return self._denormalize_yuv_42x(y, uv, 1 / 2)


class converter:
    def __init__(self, kr=0.2126, kb=0.0722, depth=8, format='yuv420', upsample_mode='bilinear'):
        self.krgb = (kr, 1 - kr - kb, kb)
        self.depth = depth
        self.uv_bias = (1 << (depth - 1)) / ((1 << depth) - 1)
        match format:
            case 'yuv444':
                self.downsample = lambda x: x
                self.upsample = lambda x: x
            case 'yuv422':
                self.downsample = functools.partial(F.interpolate, scale_factor=(1, 1 / 2), mode='bilinear',
                                                    align_corners=False)
                self.upsample = functools.partial(F.interpolate, scale_factor=(1, 2), mode=upsample_mode,
                                                  align_corners=False)
            case 'yuv420':
                self.downsample = functools.partial(F.interpolate, scale_factor=(1 / 2, 1 / 2), mode='bilinear',
                                                    align_corners=False)
                self.upsample = functools.partial(F.interpolate, scale_factor=(2, 2), mode=upsample_mode,
                                                  align_corners=False)

    def rgb2yuv(self, x: torch.Tensor):
        kr, kg, kb = self.krgb

        r, g, b = torch.chunk(x, 3, 1)
        y = kr * r + kg * g + kb * b
        u = (y - b) / (kb - 1) / 2 + self.uv_bias
        v = (y - r) / (kr - 1) / 2 + self.uv_bias
        uv = torch.cat((u, v), dim=1)
        return y, self.downsample(uv)

    def yuv2rgb(self, y: torch.Tensor, uv: torch.Tensor):
        kr, kg, kb = self.krgb

        uv = self.upsample(uv - self.uv_bias)
        u, v = torch.chunk(uv, 2, 1)
        r = y + 2 * (1 - kr) * v
        b = y + 2 * (1 - kb) * u
        g = y - 2 * (1 - kr) * kr * v - 2 * (1 - kb) * kb * u
        return torch.cat((r, g, b), dim=1)
