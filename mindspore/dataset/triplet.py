import pathlib
import random
from typing import List

import cv2
import mindspore as ms
from mindspore import dataset
from mindspore.dataset import vision, transforms
import numpy as np
from PIL import Image, ImageFilter


class ImageTripletGenerator:
    def __init__(self, index_file, patch_size, scale_factor, augment, seed=0):
        self.dataset_base = pathlib.Path(index_file).parent
        self.triplets = [i for i in open(index_file, 'r', encoding='utf-8').read().split('\n')
                         if i if not i.startswith('#')]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.rand = random.Random(seed)

    def _load_triplet(self, path):
        path = self.dataset_base / "sequences" / path
        images = [Image.open(path / f"im{i + 1}.png") for i in range(3)]
        if not (images[0].size == images[1].size and images[0].size == images[2].size):
            raise ValueError("triplet has different dimensions")
        return images

    def _prepare_images(self, images: List[Image.Image]):
        w, h = images[0].size
        f = self.scale_factor
        s = self.patch_size * f
        dh, dw = self.rand.randrange(0, h - s, 2) * f, self.rand.randrange(0, w - s, 2) * f
        images = [i.crop((dw, dh, dw + s, dh + s)) for i in images]
        return images

    trans_groups = {
        'none': [None],
        'rotate': [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        'mirror': [None, Image.FLIP_LEFT_RIGHT],
        'flip': [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180],
        'all': [None] + [e.value for e in Image.Transpose],
    }

    trans_names = [e.name for e in Image.Transpose]

    def _augment_images(self, images: List[Image.Image], trans_mode='all'):
        trans_action = 'none'
        trans_op = self.rand.choice(self.trans_groups[trans_mode])
        if trans_op is not None:
            images = [i.transpose(trans_op) for i in images]
            trans_action = self.trans_names[trans_op]
        return images, trans_action

    scale_filters = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

    def _scale_images(self, images: List[Image.Image]):
        f = self.scale_factor
        return [i.resize((i.width // f, i.height // f), self.rand.choice(self.scale_filters)) for i in images]

    def _degrade_images(self, images: List[Image.Image]):
        degrade_action = None
        decision = self.rand.randrange(4)
        if decision == 1:
            degrade_action = 'box'
            arr = [np.array(Image.blend(j, j.copy().filter(ImageFilter.BoxBlur(1)), 0.5)) for j in images]
        elif decision == 2:
            degrade_action = 'gaussian'
            radius = self.rand.random() * 2
            arr = [np.array(j.filter(ImageFilter.GaussianBlur(radius))) for j in images]
        elif decision == 3:
            degrade_action = 'halo'
            radius = 1 + self.rand.random() * 2
            modulation = 0.1 + radius * 0.3
            contour = [np.array(i.copy().filter(ImageFilter.CONTOUR).filter(ImageFilter.GaussianBlur(radius)))
                       for i in images]
            arr = [cv2.addWeighted(np.array(i), 1, j, modulation, 0) for i, j in zip(images, contour)]
        else:
            arr = [np.array(i) for i in images]

        return arr, degrade_action

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self._load_triplet(self.triplets[idx])
        triplet = self._prepare_images(triplet)  # crop to requested size
        original, _ = self._augment_images(triplet)  # flip and rotates
        lf1 = original[1]
        lf1 = np.array(lf1.resize((lf1.width // self.scale_factor, lf1.height // self.scale_factor), Image.LANCZOS))
        degraded, _ = self._degrade_images(self._scale_images([original[0], original[2]]))
        degraded.insert(1, lf1)
        return (*original, *degraded)


def ImageTripletDataset(index_file, patch_size, scale_factor, augment, normalizer):
    ds = dataset.GeneratorDataset(
        ImageTripletGenerator(index_file, patch_size, scale_factor, augment),
        column_names=[f'{s}{i}' for s in ('h', 'l') for i in range(3)],
    )

    mean, std = normalizer.rgb_dist()

    for col in [f'{s}{i}' for s in ('h', 'l') for i in range(3)]:
        ds = ds.map([
            transforms.TypeCast(ms.float32),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean, std, is_hwc=True),
            vision.HWC2CHW()
        ], col)

    return ds
