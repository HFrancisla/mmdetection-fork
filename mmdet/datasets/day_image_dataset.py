# Copyright (c) OpenMMLab. All rights reserved.
"""Unlabeled daytime images for the D_V6 PatchGAN real domain."""

import random
from pathlib import Path
from typing import Sequence

import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset


class DayImageDataset(Dataset):
    """Read unlabeled BGR images and apply detector-like size augmentation.

    The returned uint8 tensors are BGR in the 0..255 range. The model's
    DetDataPreprocessor then performs the same BGR-to-RGB conversion and
    ImageNet normalization used for the labeled night batch.
    """

    IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self,
                 image_root: str,
                 max_side_choices: Sequence[int] = (320, 608),
                 horizontal_flip_prob: float = 0.5):
        if not image_root:
            raise ValueError(
                'Set train_cfg.day_root to a directory of daytime images.')
        self.image_root = Path(image_root).expanduser()
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f'Daytime image directory does not exist: {self.image_root}')

        self.image_paths = sorted(
            path for path in self.image_root.rglob('*')
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES)
        if not self.image_paths:
            raise ValueError(
                f'No supported images found under {self.image_root}')

        self.max_side_choices = tuple(int(size) for size in max_side_choices)
        if not self.max_side_choices or any(size <= 0
                                            for size in self.max_side_choices):
            raise ValueError('max_side_choices must contain positive sizes')
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError('horizontal_flip_prob must be between 0 and 1')
        self.horizontal_flip_prob = float(horizontal_flip_prob)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        image = mmcv.imread(str(path), flag='color', channel_order='bgr')
        if image is None:
            raise OSError(f'Unable to read daytime image: {path}')

        height, width = image.shape[:2]
        max_side = random.choice(self.max_side_choices)
        scale = max_side / max(height, width)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if (new_width, new_height) != (width, height):
            image = mmcv.imresize(
                image, (new_width, new_height), interpolation='bilinear')

        if random.random() < self.horizontal_flip_prob:
            image = np.flip(image, axis=1)

        # Keep BGR ordering here; DetDataPreprocessor converts it to RGB.
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
