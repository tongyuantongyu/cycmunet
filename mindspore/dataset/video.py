import bisect
import collections
import functools
import itertools
import pathlib
import random
import time
from typing import List, Tuple

import av
import av.logging
import cv2
import numpy as np
import mindspore as ms
from mindspore import dataset
from mindspore.dataset import vision, transforms


av.logging.set_level(av.logging.FATAL)

perf_debug = False


class Video:
    def __init__(self, file, kf):
        self.container = av.open(file)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.at = 0
        self.kf = kf

    def get_frames(self, pts, n=1):
        frames = []
        if bisect.bisect_left(self.kf, pts) != bisect.bisect_left(self.kf, self.at) or pts <= self.at:
            self.container.seek(pts, stream=self.stream)
        found = False
        first = True
        if perf_debug:
            print(f'Seek {pts} done at {time.perf_counter()}')
        for frame in self.container.decode(video=0):
            if first:
                if perf_debug:
                    print(f'Search {pts} from {frame.pts} at {time.perf_counter()}')
                first = False
            if not found and frame.pts != pts:
                continue
            found = True
            if perf_debug:
                print(f'Found {frame.pts} at {time.perf_counter()}')
            self.at = frame.pts
            yuv = frame.to_ndarray()
            h, w = frame.height, frame.width
            y, uv = yuv[:h, :].reshape(1, h, w), yuv[h:, :].reshape(2, h // 2, w // 2)
            frames.append((y, uv))
            if len(frames) == n:
                return frames
        raise ValueError("unexpected end")

    def __del__(self):
        self.container.close()


video_info = collections.namedtuple('video_info', [
    'org',
    'deg',
    'frames',
    'pts_org',
    'pts_deg',
    'key_org',
    'key_deg'
])


def flatten_once(it):
    return itertools.chain.from_iterable(it)


class VideoFrameGenerator:
    def __init__(self, index_file, patch_size, scale_factor, augment, seed=0):
        self.dataset_base = pathlib.PurePath(index_file).parent
        index_lines = [i for i in open(index_file, 'r', encoding='utf-8').read().split('\n')
                       if i if not i.startswith('#')]
        files = [tuple(i.split(',')) for i in index_lines]
        self.files = []
        self.indexes = []
        for org, deg, frames, pts_org, pts_deg, key_org, key_deg in files:
            info = video_info(
                org,
                deg,
                int(frames),
                tuple(int(i) for i in pts_org.split(' ')),
                tuple(int(i) for i in pts_deg.split(' ')),
                tuple(int(i) for i in key_org.split(' ')),
                tuple(int(i) for i in key_deg.split(' ')),
            )
            self.files.append(info)
            self.indexes.append(info.frames)
        self.indexes = list(itertools.accumulate(self.indexes))
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.rand = random.Random(seed)

    @functools.lru_cache(2)
    def get_video(self, v_idx):
        info = self.files[v_idx]
        return Video(str(self.dataset_base / info.org), info.key_org), \
            Video(str(self.dataset_base / info.deg), info.key_deg), info.pts_org, info.pts_deg

    def _augment_frame(self, org: List[Tuple[np.ndarray]], deg: List[Tuple[np.ndarray]]):
        if self.rand.random() > 0.5:
            org = [(y[..., ::-1].copy(), uv[..., ::-1].copy()) for y, uv in org]
            deg = [(y[..., ::-1].copy(), uv[..., ::-1].copy()) for y, uv in deg]
        return org, deg

    def _prepare_frame(self, org: List[Tuple[np.ndarray]], deg: List[Tuple[np.ndarray]]):
        _, h, w = deg[0][0].shape
        sw, sh = self.patch_size
        sh_uv, sw_uv = sh // 2, sw // 2
        dh, dw = self.rand.randrange(0, h - sh, 2), self.rand.randrange(0, w - sw, 2)
        dh_uv, dw_uv = dh // 2, dw // 2
        deg = [(y[:, dh:dh+sh, dw:dw+sw], uv[:, dh_uv:dh_uv+sh_uv, dw_uv:dw_uv+sw_uv]) for y, uv in deg]
        f = self.scale_factor
        size, size_uv = (sw, sh), (sw_uv, sh_uv)
        sh, sw, sh_uv, sw_uv = sh * f, sw * f, sh_uv * f, sw_uv * f
        dh, dw, dh_uv, dw_uv = dh * f, dw * f, dh_uv * f, dw_uv * f
        org = [(y[:, dh:dh+sh, dw:dw+sw], uv[:, dh_uv:dh_uv+sh_uv, dw_uv:dw_uv+sw_uv]) for y, uv in org]

        deg1_y = cv2.resize(org[1][0][0], size, interpolation=cv2.INTER_LANCZOS4)
        deg1_u = cv2.resize(org[1][1][0], size_uv, interpolation=cv2.INTER_LANCZOS4)
        deg1_v = cv2.resize(org[1][1][1], size_uv, interpolation=cv2.INTER_LANCZOS4)
        deg.insert(1, (deg1_y.reshape((1, *size[::-1])), np.stack((deg1_u, deg1_v)).reshape((2, *size_uv[::-1]))))
        return org, deg

    def __len__(self):
        return self.indexes[-1]

    def __getitem__(self, idx):
        start = time.perf_counter()
        v_idx = bisect.bisect_right(self.indexes, idx)
        f_idx = idx if v_idx == 0 else idx - self.indexes[v_idx - 1]
        org, deg, pts_org, pts_deg = self.get_video(v_idx)
        org_frames = org.get_frames(pts_org[f_idx], 3)
        deg_frames = deg.get_frames(pts_deg[f_idx], 3)
        deg_frames.pop(1)
        org_frames, deg_frames = self._prepare_frame(org_frames, deg_frames)
        if self.augment:
            org_frames, deg_frames = self._augment_frame(org_frames, deg_frames)
        ret = (*flatten_once(org_frames), *flatten_once(deg_frames))
        if perf_debug:
            print(f'Prepared data {idx}, in {time.perf_counter() - start}')
        return ret


def VideoFrameDataset(index_file, patch_size, scale_factor, augment, normalizer):
    ds = dataset.GeneratorDataset(
        VideoFrameGenerator(index_file, patch_size, scale_factor, augment),
        column_names=[f'{s}{i}_{p}' for s in ('h', 'l') for i in range(3) for p in ('y', 'uv')],
        shuffle=False,
        python_multiprocessing=True,
        num_parallel_workers=3,
    )

    mean, std = normalizer.yuv_dist()

    for col in [f'{s}{i}' for s in ('h', 'l') for i in range(3)]:
        ds = ds.map([
            transforms.TypeCast(ms.float32),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean[:1], std[:1], is_hwc=False)
        ], col + '_y')
        ds = ds.map([
            transforms.TypeCast(ms.float32),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean[1:], std[1:], is_hwc=False)
        ], col + '_uv')

    return ds
