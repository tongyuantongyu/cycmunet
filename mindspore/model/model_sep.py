from collections.abc import Iterable

import mindspore as ms
from mindspore import nn, ops

from .deform_conv import DCN_sep_compat

# half_pixel not supported on cpu now. use align_corners for test temporarily.
coordinate_transformation_mode = 'half_pixel'
# coordinate_transformation_mode = 'align_corners'


def Conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           has_bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     pad_mode='pad', padding=padding, dilation=dilation, has_bias=has_bias)


def Conv2dTranspose(in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=1):
    return nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, pad_mode='same', dilation=dilation, has_bias=True)


def float_tuple(*tuple_):
    tuple_ = tuple_[0] if isinstance(tuple_[0], Iterable) else tuple_
    return tuple(float(i) for i in tuple_)


def transpose(mat):
    count = len(mat[0])
    ret = [[] for _ in range(count)]
    for i in mat:
        for j in range(count):
            ret[j].append(i[j])
    return ret


def cat_simp(elements, axis=1):
    if len(elements) == 1:
        return elements[0]
    else:
        return ms.ops.concat(elements, axis=axis)


class CatConv(nn.Cell):
    def __init__(self, in_channels, in_count, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.convs = nn.CellList(
            [Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, i == 0) for i in
             range(in_count)])

    def construct(self, *inputs):
        output = None
        for idx, inp in enumerate(inputs):
            if output is None:
                output = self.convs[idx](inp)
            else:
                output += self.convs[idx](inp)
        return output


class Pro_align(nn.Cell):
    def __init__(self, args):
        super(Pro_align, self).__init__()
        self.args = args
        self.nf = self.args.nf

        self.conv1x1 = CatConv(self.nf, 3, self.nf, 1, 1, 0)
        self.conv3x3 = Conv2d(self.nf, self.nf, 3, 1, 1)
        self.conv1_3x3 = CatConv(self.nf, 2, self.nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(alpha=0.1)

    def construct(self, l1, l2, l3):
        r1 = self.lrelu(self.conv3x3(l1))
        r2 = self.lrelu(self.conv3x3(l2))
        r3 = self.lrelu(self.conv3x3(l3))
        fuse = self.lrelu(self.conv1x1(r1, r2, r3))
        r1 = self.lrelu(self.conv1_3x3(r1, fuse))
        r2 = self.lrelu(self.conv1_3x3(r2, fuse))
        r3 = self.lrelu(self.conv1_3x3(r3, fuse))
        return l1 + r1, l2 + r2, l3 + r3


class SR(nn.Cell):
    def __init__(self, args):
        super(SR, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (1, 1, self.args.upscale_factor, self.args.upscale_factor)
        self.Pro_align = Pro_align(args)
        self.conv1x1 = Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(alpha=0.1)

    def upsample(self, x):
        x = ops.interpolate(x, scales=float_tuple(self.factor), mode='bilinear',
                            coordinate_transformation_mode=coordinate_transformation_mode)
        return self.lrelu(self.conv1x1(x))

    def construct(self, l1, l2, l3):
        l1, l2, l3 = self.Pro_align(l1, l2, l3)
        return tuple(self.upsample(i) for i in (l1, l2, l3))


class DR(nn.Cell):
    def __init__(self, args):
        super(DR, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.factor = (1, 1, 1 / self.args.upscale_factor, 1 / self.args.upscale_factor)
        self.Pro_align = Pro_align(args)
        self.conv = Conv2d(self.nf, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(alpha=0.1)

    def downsample(self, x):
        x = ops.interpolate(x, scales=float_tuple(self.factor), mode='bilinear',
                            coordinate_transformation_mode=coordinate_transformation_mode)
        return self.lrelu(self.conv(x))

    def construct(self, l1, l2, l3):
        l1 = self.downsample(l1)
        l2 = self.downsample(l2)
        l3 = self.downsample(l3)
        return self.Pro_align(l1, l2, l3)


class Up_projection(nn.Cell):
    def __init__(self, args):
        super(Up_projection, self).__init__()
        self.args = args
        self.SR = SR(args)
        self.DR = DR(args)
        self.SR1 = SR(args)

    def construct(self, l1, l2, l3):
        h1, h2, h3 = self.SR(l1, l2, l3)
        d1, d2, d3 = self.DR(h1, h2, h3)
        r1, r2, r3 = d1 - l1, d2 - l2, d3 - l3
        s1, s2, s3 = self.SR1(r1, r2, r3)
        return h1 + s1, h2 + s3, h3 + s3


class Down_projection(nn.Cell):
    def __init__(self, args):
        super(Down_projection, self).__init__()
        self.args = args
        self.SR = SR(args)
        self.DR = DR(args)
        self.DR1 = DR(args)

    def construct(self, h1, h2, h3):
        l1, l2, l3 = self.DR(h1, h2, h3)
        s1, s2, s3 = self.SR(l1, l2, l3)
        r1, r2, r3 = s1 - h1, s2 - h2, s3 - h3
        d1, d2, d3 = self.DR1(r1, r2, r3)
        return l1 + d1, l2 + d2, l3 + d3


class FusionPyramidLayer(nn.Cell):
    def __init__(self, args, first_layer: bool):
        super(FusionPyramidLayer, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups

        self.offset_conv1 = CatConv(self.nf, 2, self.nf, 3, 1, 1)
        self.offset_conv3 = Conv2d(self.nf, self.nf, 3, 1, 1)
        self.dcnpack = DCN_sep_compat(self.nf, self.nf, self.nf, 3, stride=1, padding=1, dilation=1,
                                      deformable_groups=self.groups)
        self.lrelu = nn.LeakyReLU(alpha=0.1)

        if not first_layer:
            self.offset_conv2 = CatConv(self.nf, 2, self.nf, 3, 1, 1)
            self.fea_conv = CatConv(self.nf, 2, self.nf, 3, 1, 1)

    def construct(self, current_sources, last_offset, last_feature):
        offset = self.lrelu(self.offset_conv1(*current_sources))
        if last_offset is not None:
            last_offset = ops.interpolate(last_offset, scales=float_tuple(1, 1, 2, 2), mode='bilinear',
                                          coordinate_transformation_mode=coordinate_transformation_mode)
            offset = self.lrelu(self.offset_conv2(offset, last_offset * 2))
        offset = self.lrelu(self.offset_conv3(offset))
        feature = self.dcnpack(current_sources[0], offset)
        if last_feature is not None:
            last_feature = ops.interpolate(last_feature, scales=float_tuple(1, 1, 2, 2), mode='bilinear',
                                           coordinate_transformation_mode=coordinate_transformation_mode)
            feature = self.fea_conv(feature, last_feature)
        feature = self.lrelu(feature)
        return offset, feature


class ResidualBlock_noBN(nn.Cell):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = Conv2d(nf, nf, 3, 1, 1)
        self.relu = ops.ReLU()

        # TODO: initialization
        # initialize_weights([self.conv1, self.conv2], 0.1)

    def construct(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class cycmunet_head(nn.Cell):
    def __init__(self, args):
        super(cycmunet_head, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.lrelu = nn.LeakyReLU(alpha=0.1)
        if self.args.format == 'rgb':
            self.conv_first = Conv2d(3, self.nf, 3, 1, 1)
            self._construct = self.construct_rgb
        elif self.args.format == 'yuv444':
            self.conv_first = Conv2d(3, self.nf, 3, 1, 1)
            self._construct = self.construct_yuv444
        elif self.args.format == 'yuv422':
            self.conv_first_y = Conv2d(1, self.nf, 3, 1, 1)
            self.conv_up = Conv2dTranspose(2, self.nf, (1, 3), (1, 2))
            self._construct = self.construct_yuv42x
        elif self.args.format == 'yuv420':
            self.conv_first_y = Conv2d(1, self.nf, 3, 1, 1)
            self.conv_up = Conv2dTranspose(2, self.nf, 3, 2)
            self._construct = self.construct_yuv42x
        else:
            raise ValueError(f'unknown input pixel format: {self.args.format}')

    def construct_rgb(self, x):
        x = self.lrelu(self.conv_first(x))
        return x

    def construct_yuv444(self, yuv):
        x = ms.ops.concat(yuv, axis=1)
        x = self.lrelu(self.conv_first(x))
        return x

    def construct_yuv42x(self, yuv):
        y, uv = yuv
        y = self.conv_first_y(y)
        uv = self.conv_up(uv)
        x = self.lrelu(y + uv)
        return x

    def construct(self, *args):
        return self._construct(*args)


class cycmunet_feature_extract(nn.Cell):
    def __init__(self, args):
        super(cycmunet_feature_extract, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups
        self.layers = self.args.layers
        self.front_RBs = 5

        self.feature_extraction = nn.SequentialCell(*(ResidualBlock_noBN(nf=self.nf) for _ in range(self.front_RBs)))
        self.fea_conv1s = nn.CellList([Conv2d(self.nf, self.nf, 3, 2, 1) for _ in range(self.layers - 1)])
        self.fea_conv2s = nn.CellList([Conv2d(self.nf, self.nf, 3, 1, 1) for _ in range(self.layers - 1)])
        self.lrelu = nn.LeakyReLU(alpha=0.1)

    def construct(self, x):
        features = [self.feature_extraction(x)]
        for i in range(self.layers - 1):
            feature = features[-1]
            feature = self.lrelu(self.fea_conv1s[i](feature))
            feature = self.lrelu(self.fea_conv2s[i](feature))
            features.append(feature)
        return tuple(features[::-1])  # lowest dimension layer at first


class cycmunet_feature_fusion(nn.Cell):
    def __init__(self, args):
        super(cycmunet_feature_fusion, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.groups = self.args.groups
        self.layers = self.args.layers

        # from small to big.
        self.modules12 = nn.CellList([FusionPyramidLayer(args, i == 0) for i in range(self.layers)])
        self.modules21 = nn.CellList([FusionPyramidLayer(args, i == 0) for i in range(self.layers)])

        self.fusion = CatConv(self.nf, 2, self.nf, 1, 1)

        self.lrelu = nn.LeakyReLU(alpha=0.1)

    @staticmethod
    def fuse_features(modules, f1, f2):
        offset, feature = None, None
        for idx, sources in enumerate(zip(f1, f2)):
            offset, feature = modules[idx](sources, offset, feature)
        return feature

    def construct(self, f1, f2):
        feature1 = self.fuse_features(self.modules12, f1, f2)
        feature2 = self.fuse_features(self.modules21, f2, f1)
        fused_feature = self.fusion(feature1, feature2)
        return fused_feature, None  # TODO detect scene change here


class cycmunet_mutual_cycle(nn.Cell):
    def __init__(self, args):
        super(cycmunet_mutual_cycle, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.cycle_count = self.args.cycle_count
        self.all_frames = self.args.all_frames

        self.merge = nn.CellList([CatConv(64, i + 1, 64, 1, 1, 0) for i in range(self.cycle_count)])
        self.merge1 = nn.CellList([CatConv(64, i + 1, 64, 1, 1, 0) for i in range(self.cycle_count)])

        self.down = nn.CellList([Down_projection(args) for _ in range(self.cycle_count)])
        self.up = nn.CellList([Up_projection(args) for _ in range(self.cycle_count + 1)])

        self.conv = CatConv(self.nf, 2 * self.cycle_count + 1, self.nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(alpha=0.1)

    def construct(self, lf0, lf1, lf2):
        l_out, h_out = [(lf0, lf1, lf2)], []
        l_feats = []
        for j1 in range(self.cycle_count):
            l_feats = tuple(self.lrelu(self.merge[j1](*frame_outs)) for frame_outs in transpose(l_out))
            h_feat = self.up[j1](*l_feats)
            h_out.append(h_feat)
            h_feats = tuple(self.lrelu(self.merge1[j1](*frame_outs)) for frame_outs in transpose(h_out))
            l_feat = self.down[j1](*h_feats)
            l_out.append(l_feat)

        lf_out, hf_out = [l_out[-1]], []
        for j2 in range(self.cycle_count):
            l_feats = tuple(self.lrelu(self.merge[j2](*frame_outs)) for frame_outs in transpose(lf_out))
            h_feat = self.up[j2](*l_feats)
            hf_out.append(h_feat)
            l_feat = self.down[j2](*h_feat)
            lf_out.append(l_feat)
        hf_out.append(self.up[self.cycle_count](*l_feats))

        if self.all_frames:
            h_outs = transpose(h_out + hf_out)  # packed 3 frames
            _, l1_out, _ = transpose(l_out + lf_out[1:])

            h_outs = list(self.lrelu(self.conv(*h_frame)) for h_frame in h_outs)
            l1_out = self.lrelu(self.conv(*l1_out))
            return tuple(h_outs + [l1_out])
        else:
            h1_out, h2_out, _ = transpose(h_out + hf_out)
            h1_out = self.lrelu(self.conv(*h1_out))
            h2_out = self.lrelu(self.conv(*h2_out))

            return h1_out, h2_out


class cycmunet_feature_recon(nn.Cell):
    def __init__(self, args):
        super(cycmunet_feature_recon, self).__init__()
        self.args = args
        self.nf = self.args.nf
        self.back_RBs = 40

        self.recon_trunk = nn.SequentialCell(*(ResidualBlock_noBN(nf=self.nf) for _ in range(self.back_RBs)))

    def construct(self, x):
        out = self.recon_trunk(x)
        return out


class cycmunet_tail(nn.Cell):
    def __init__(self, args):
        super(cycmunet_tail, self).__init__()
        self.args = args
        self.nf = self.args.nf
        if self.args.format == 'rgb':
            self.conv_last2 = Conv2d(self.nf, 3, 3, 1, 1)
            self._construct = self.construct_rgb
        elif self.args.format == 'yuv444':
            self.conv_last2 = Conv2d(self.nf, 3, 3, 1, 1)
            self._construct = self.construct_yuv444
        elif self.args.format == 'yuv422':
            self.conv_last_y = Conv2d(self.nf, 1, 3, 1, 1)
            self.conv_last_uv = Conv2d(self.nf, 2, (1, 3), (1, 2), (0, 0, 1, 1))
            self._construct = self.construct_yuv42x
        elif self.args.format == 'yuv420':
            self.conv_last_y = Conv2d(self.nf, 1, 3, 1, 1)
            self.conv_last_uv = Conv2d(self.nf, 2, 3, 2, 1)
            self._construct = self.construct_yuv42x
        else:
            raise ValueError(f'unknown input pixel format: {self.args.format}')

    def construct_rgb(self, x):
        out = self.conv_last2(x)
        return out,

    def construct_yuv444(self, x):
        out = self.conv_last2(x)
        return out[:, :1, ...], out[:, 1:, ...]

    def construct_yuv42x(self, x):
        y = self.conv_last_y(x)
        uv = self.conv_last_uv(x)
        return y, uv

    def construct(self, *args):
        return self._construct(*args)


class CycMuNet(nn.Cell):
    def __init__(self, args):
        super(CycMuNet, self).__init__()
        self.args = args
        self.factor = (1, 1, self.args.upscale_factor, self.args.upscale_factor)
        self.upsample_mode = 'bilinear'
        self.batch_mode = self.args.batch_mode
        self.stop_at_conf = self.args.stop_at_conf
        self.all_frames = self.args.all_frames

        self.head = cycmunet_head(args)
        self.fe = cycmunet_feature_extract(args)
        self.ff = cycmunet_feature_fusion(args)
        self.mu = cycmunet_mutual_cycle(args)
        self.fr = cycmunet_feature_recon(args)
        self.tail = cycmunet_tail(args)

    def merge_hf(self, lf, hf):
        return ops.interpolate(lf, scales=float_tuple(self.factor), mode='bilinear',
                               coordinate_transformation_mode=coordinate_transformation_mode) + hf

    def mu_fr(self, *lf):
        mu_out = self.mu(*lf)
        if self.all_frames:
            # *hf, lf1 = mu_out
            hf, lf1 = mu_out[:-1], mu_out[-1]
            hf = tuple(self.merge_hf(l, self.fr(h)) for l, h in zip(lf, hf))
            outs = list(hf) + [lf1]
            outs = tuple(self.tail(i) for i in outs)
        else:
            outs = tuple(self.tail(self.merge_hf(l, self.fr(h))) for l, h in zip(lf, mu_out))
        return outs

    def construct_batch(self, f0, f2):
        lf0s = self.fe(self.head(f0))
        lf2s = self.fe(self.head(f2))
        lf0, lf2 = lf0s[self.layers - 1], lf2s[self.layers - 1]
        lf1, _ = self.ff(lf0s, lf2s)
        if self.stop_at_conf:  # TODO detect frame difference and exit if too big
            return
        return self.mu_fr(lf0, lf1, lf2)

    def construct_sequence(self, x_or_yuv):
        x = self.head(x_or_yuv)
        ls = self.fe(x)
        lf0, lf2 = ls[2][:-1], ls[2][1:]
        lf1, _ = self.ff([layer[:-1] for layer in ls], [layer[1:] for layer in ls])
        return self.mu_fr(lf0, lf1, lf2)

    def construct(self, f0, f2=None):
        if self.batch_mode == 'batch':
            outs = self.construct_batch(f0, f2)
        elif self.batch_mode == 'sequence':
            outs = self.construct_sequence(f0)
        else:
            raise ValueError(f"Invalid batch_mode: {self.args.batch_mode}")
        return outs
