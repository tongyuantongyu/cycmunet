from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import cat_conv
from .part import ResidualBlock_noBN, DCN_sep, Down_projection, Up_projection


class head(nn.Module):
    def __init__(self, args):
        super(head, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        match self.args.format:
            case 'rgb':
                self.conv_first = nn.Conv2d(3, self.nf, 3, 1, 1)
                self.forward = self.forward_rgb
            case 'yuv444':
                self.conv_first = nn.Conv2d(3, self.nf, 3, 1, 1)
                self.forward = self.forward_yuv444
            case 'yuv422':
                self.conv_first_y = nn.Conv2d(1, self.nf, 3, 1, 1)
                self.conv_up = nn.ConvTranspose2d(2, self.nf, (1, 3), (1, 2), (0, 1), (0, 1))
                self.forward = self.forward_yuv42x
            case 'yuv420':
                self.conv_first_y = nn.Conv2d(1, self.nf, 3, 1, 1)
                self.conv_up = nn.ConvTranspose2d(2, self.nf, 3, 2, 1, 1)
                self.forward = self.forward_yuv42x
            case unk:
                raise ValueError(f'unknown input pixel format: {unk}')

    def forward_rgb(self, x: torch.Tensor):
        x = self.lrelu(self.conv_first(x))
        return x

    def forward_yuv444(self, yuv: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(yuv, dim=1)
        x = self.lrelu(self.conv_first(x))
        return x

    def forward_yuv42x(self, yuv: Tuple[torch.Tensor, torch.Tensor]):
        y, uv = yuv
        y = self.conv_first_y(y)
        uv = self.conv_up(uv)
        x = self.lrelu(y + uv)
        return x


class feature_extract(nn.Module):
    def __init__(self, args):
        super(feature_extract, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups
        self.layers = self.args.layers
        self.front_RBs = 5

        self.feature_extraction = nn.Sequential(*(ResidualBlock_noBN(nf=self.nf) for _ in range(self.front_RBs)))
        self.fea_conv1s = nn.ModuleList(nn.Conv2d(self.nf, self.nf, 3, 2, 1, bias=True) for _ in range(self.layers - 1))
        self.fea_conv2s = nn.ModuleList(nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True) for _ in range(self.layers - 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor):
        features = [self.feature_extraction(x)]
        for i in range(self.layers - 1):
            feature = features[-1]
            feature = self.lrelu(self.fea_conv1s[i](feature))
            feature = self.lrelu(self.fea_conv2s[i](feature))
            features.append(feature)
        return tuple(features[::-1])  # lowest dimension layer at first


class FusionPyramidLayer(nn.Module):
    def __init__(self, args, first_layer: bool):
        super(FusionPyramidLayer, self).__init__()
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
            offset = self.lrelu(cat_conv(self.offset_conv2, (offset, last_offset * 2)))
        offset = self.lrelu(self.offset_conv3(offset))
        feature = self.dcnpack(current_sources[0], offset)
        if last_feature is not None:
            last_feature = F.interpolate(last_feature, scale_factor=2, mode='bilinear', align_corners=False)
            feature = cat_conv(self.fea_conv, (feature, last_feature))
        feature = self.lrelu(feature)
        return offset, feature


class feature_fusion(nn.Module):
    def __init__(self, args):
        super(feature_fusion, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups
        self.layers = self.args.layers

        # from small to big.
        self.modules12 = nn.ModuleList(FusionPyramidLayer(args, i == 0) for i in range(self.layers))
        self.modules21 = nn.ModuleList(FusionPyramidLayer(args, i == 0) for i in range(self.layers))

        self.fusion = nn.Conv2d(2 * self.nf, self.nf, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    @staticmethod
    def fuse_features(modules, f1, f2):
        offset, feature = None, None
        for idx, sources in enumerate(zip(f1, f2)):
            offset, feature = modules[idx](sources, offset, feature)
        return feature

    def forward(self, f1, f2):
        feature1 = self.fuse_features(self.modules12, f1, f2)
        feature2 = self.fuse_features(self.modules21, f2, f1)
        fused_feature = cat_conv(self.fusion, (feature1, feature2))
        return fused_feature, None


class mutual_cycle(nn.Module):
    def __init__(self, args):
        super(mutual_cycle, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.cycle_count = self.args.cycle_count

        self.merge = nn.ModuleList(nn.Conv2d(64 * (i + 1), 64, 1, 1, 0) for i in range(self.cycle_count))
        self.merge1 = nn.ModuleList(nn.Conv2d(64 * (i + 1), 64, 1, 1, 0) for i in range(self.cycle_count))

        self.down = nn.ModuleList(Down_projection(args) for _ in range(self.cycle_count))
        self.up = nn.ModuleList(Up_projection(args) for _ in range(self.cycle_count + 1))

        self.conv = nn.Conv2d(self.nf * (2 * self.cycle_count + 1), self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lf0, lf1, lf2, all_frames=False):
        assert self.cycle_count > 0

        l_out, h_out = [(lf0, lf1, lf2)], []
        for j in range(self.cycle_count):
            l_feats = tuple(self.lrelu(cat_conv(self.merge[j], frame_outs)) for frame_outs in zip(*l_out))
            h_feat = self.up[j](*l_feats)
            h_out.append(h_feat)
            h_feats = tuple(self.lrelu(cat_conv(self.merge1[j], frame_outs)) for frame_outs in zip(*h_out))
            l_feat = self.down[j](*h_feats)
            l_out.append(l_feat)

        lf_out, hf_out = [l_out[-1]], []
        for j in range(self.cycle_count):
            l_feats = tuple(self.lrelu(cat_conv(self.merge[j], frame_outs)) for frame_outs in zip(*lf_out))
            h_feat = self.up[j](*l_feats)
            hf_out.append(h_feat)
            l_feat = self.down[j](*h_feat)
            lf_out.append(l_feat)
        hf_out.append(self.up[self.cycle_count](*l_feats))

        if all_frames:
            h_outs = zip(*h_out, *hf_out)  # packed 3 frames
            _, l1_out, _ = zip(*l_out, *lf_out[1:])

            h_outs = tuple(self.lrelu(cat_conv(self.conv, h_frame)) for h_frame in h_outs)
            l1_out = self.lrelu(cat_conv(self.conv, l1_out))
            return *h_outs, l1_out
        else:
            h1_out, h2_out, _ = zip(*h_out, *hf_out)
            h1_out = self.lrelu(cat_conv(self.conv, h1_out))
            h2_out = self.lrelu(cat_conv(self.conv, h2_out))

            return h1_out, h2_out


class feature_recon(nn.Module):
    def __init__(self, args):
        super(feature_recon, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.back_RBs = 40
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)

        self.recon_trunk = nn.Sequential(*(ResidualBlock_noBN(nf=self.nf) for _ in range(self.back_RBs)))

    def forward(self, x):
        out = self.recon_trunk(x)
        return out


class tail(nn.Module):
    def __init__(self, args):
        super(tail, self).__init__()
        self.args = args
        self.nf = self.args.nf
        match self.args.format:
            case 'rgb':
                self.conv_last2 = nn.Conv2d(self.nf, 3, 3, 1, 1)
                self.forward = self.forward_rgb
            case 'yuv444':
                self.conv_last2 = nn.Conv2d(self.nf, 3, 3, 1, 1)
                self.forward = self.forward_yuv444
            case 'yuv422':
                self.conv_last_y = nn.Conv2d(self.nf, 1, 3, 1, 1)
                self.conv_last_uv = nn.Conv2d(self.nf, 2, (1, 3), (1, 2), (0, 1))
                self.forward = self.forward_yuv42x
            case 'yuv420':
                self.conv_last_y = nn.Conv2d(self.nf, 1, 3, 1, 1)
                self.conv_last_uv = nn.Conv2d(self.nf, 2, 3, 2, 1)
                self.forward = self.forward_yuv42x
            case unk:
                raise ValueError(f'unknown input pixel format: {unk}')

    def forward_rgb(self, x):
        out = self.conv_last2(x)
        return out,

    def forward_yuv444(self, x):
        out = self.conv_last2(x)
        return out[:, :1, ...], out[:, 1:, ...]

    def forward_yuv42x(self, x):
        y = self.conv_last_y(x)
        uv = self.conv_last_uv(x)
        return y, uv
