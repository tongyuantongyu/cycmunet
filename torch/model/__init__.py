from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import head, feature_extract, feature_fusion, mutual_cycle, feature_recon, tail
from .util import use_fold_catconv

RGBOrYUV = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class CycMuNet(nn.Module):
    def __init__(self, args):
        super(CycMuNet, self).__init__()
        self.args = args
        self.factor = (self.args.upscale_factor, self.args.upscale_factor)
        self.upsample_mode = 'bilinear'

        self.head = head(args)
        self.fe = feature_extract(args)
        self.ff = feature_fusion(args)
        self.mu = mutual_cycle(args)
        self.fr = feature_recon(args)
        self.tail = tail(args)

    def merge_hf(self, lf, hf):
        return F.interpolate(lf, scale_factor=self.factor, mode='bilinear', align_corners=False) + hf

    def head_fe(self, x_or_yuv: RGBOrYUV):
        x = self.head(x_or_yuv)
        return self.fe(x)

    def mu_fr_tail(self, lf, all_frames):
        mu_out = self.mu(*lf, all_frames=all_frames)
        if all_frames:
            *hf, lf1 = mu_out
            lf1 = self.fr(lf1)
            hf = tuple(self.merge_hf(l, self.fr(h)) for l, h in zip(lf, hf))
            outs = tuple(self.tail(i) for i in (*hf, lf1))
        else:
            outs = tuple(self.tail(self.merge_hf(l, self.fr(h))) for l, h in zip(lf, mu_out))
        return outs

    def forward_batch(self, lf0: RGBOrYUV, lf2: RGBOrYUV, all_frames=True, stop_at_conf=False):
        lf0s, lf2s = self.head_fe(lf0), self.head_fe(lf2)
        lf1, _ = self.ff(lf0s, lf2s)
        if stop_at_conf:  # TODO detect frame difference and exit if too big
            return
        lf = (lf0s[-1], lf1, lf2s[-1])
        return self.mu_fr_tail(lf, all_frames)

    def forward_sequence(self, x_or_yuv: RGBOrYUV, all_frames=False):
        ls = self.head_fe(x_or_yuv)
        n = ls[0].shape[0]
        lf1, _ = self.ff([layer[:n - 1] for layer in ls], [layer[1:] for layer in ls])
        lf = (ls[-1][:n - 1], lf1, ls[-1][1:])
        return self.mu_fr_tail(lf, all_frames)

    # This is for symbolic tracing for sparsity
    def pseudo_forward_sparsity(self, lf0, lf1, lf2):
        hf0, *_ = self.mu(lf0, lf1, lf2, all_frames=True)
        return self.fr(hf0)

    def forward(self, lf0: RGBOrYUV, lf2: Union[RGBOrYUV, None] = None, sparsity_ex=None, /, batch_mode='batch',
                **kwargs):
        if batch_mode == '_no_use_sparsity_pseudo':
            return self.pseudo_forward_sparsity(lf0, lf2, sparsity_ex)
        if batch_mode == 'batch':
            outs = self.forward_batch(lf0, lf2, **kwargs)
        elif batch_mode == 'sequence':
            outs = self.forward_sequence(lf0, **kwargs)
        else:
            raise ValueError(f"Invalid batch_mode: {batch_mode}")
        return tuple(outs)
