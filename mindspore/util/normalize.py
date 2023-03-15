import math


class Normalizer:
    sqrt1_2 = 1 / math.sqrt(2)

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

    def rgb_dist(self):
        return self.mean, self.std

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

    def yuv_dist(self, mode='yuv420'):
        mean, std = self._yuv_dist()
        if mode == 'yuv422':
            std[1], std[2] = std[1] * self.sqrt1_2, std[2] * self.sqrt1_2
        elif mode == 'yuv420':
            std[1], std[2] = std[1] * 0.5, std[2] * 0.5
        return mean, std
