import glob
import itertools
import pathlib
import random
from typing import List

import torch.utils.data as data
import numpy as np
import torchvision.transforms
from PIL import Image, ImageFilter


class ImageSequenceDataset(data.Dataset):
    want_shuffle = True
    pix_type = 'rgb'

    def __init__(self, index_file, patch_size, scale_factor, augment, seed=0):
        self.dataset_base = pathlib.Path(index_file).parent
        self.sequences = [i for i in open(index_file, 'r', encoding='utf-8').read().split('\n')
                          if i if not i.startswith('#')]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.rand = random.Random(seed)
        self.transform = torchvision.transforms.ToTensor()

    def _load_sequence(self, path):
        path = self.dataset_base / "sequences" / path
        files = glob.glob("*.png", root_dir=path)
        assert len(files) > 1
        images = [Image.open(file) for file in files]
        if not all(i.size != images[0].size for i in images[1:]):
            raise ValueError("sequence has different dimensions")
        return images

    def _prepare_images(self, images: List[Image.Image]):
        w, h = images[0].size
        f = self.scale_factor
        sw, sh = self.patch_size
        sw, sh = sw * f, sh * f
        assert h >= sh and w >= sw
        dh, dw = self.rand.randint(0, h - sh), self.rand.randint(0, w - sw)
        images = [i.crop((dw, dh, dw + sw, dh + sh)) for i in images]
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
            percent = 0.5 + 0.5 * self.rand.random()
            images = [Image.blend(j, j.copy().filter(ImageFilter.BoxBlur(1)), percent) for j in images]
        elif decision == 2:
            degrade_action = 'gaussian'
            radius = self.rand.random()
            images = [j.filter(ImageFilter.GaussianBlur(radius)) for j in images]
        elif decision == 3:
            degrade_action = 'halo'
            percent = 0.5 + 0.5 * self.rand.random()
            images = [Image.blend(i,
                                  i.resize((i.width // 2, i.height // 2), resample=Image.LANCZOS)
                                  .resize(i.size, resample=Image.BILINEAR), percent)
                      for i in images]

        return images, degrade_action

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self._load_sequence(self.sequences[idx])
        sequence = self._prepare_images(sequence)  # crop to requested size
        original, _ = self._augment_images(sequence)  # flip and rotates
        lfs_pred = [np.array(lf.resize((lf.width // self.scale_factor, lf.height // self.scale_factor), Image.LANCZOS))
                    for lf in original[1::2]]
        lfs_deg = self._scale_images(original[::2])
        # lfs_deg, _ = self._degrade_images(lfs_deg)
        degraded = [i for i in itertools.zip_longest(lfs_deg, lfs_pred) if i is not None]
        original = [self.transform(i) for i in original]
        degraded = [self.transform(i) for i in degraded]
        return original, degraded
