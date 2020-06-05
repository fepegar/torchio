#!/usr/bin/env python

import torch
import numpy as np
import torchio
from ..utils import TorchioTestCase


class TestTransforms(TorchioTestCase):
    """Tests for all transforms."""

    def get_transform(self, channels, is_3d=True):
        landmarks_dict = {
            channel: np.linspace(0, 100, 13) for channel in channels
        }
        disp = 1 if is_3d else (0.01, 1, 1)
        elastic = torchio.RandomElasticDeformation(max_displacement=disp)
        cp_args = (9, 21, 30) if is_3d else (1, 21, 30)
        flip_axes = (0, 1, 2) if is_3d else (0, 1)
        swap_patch = (2, 3, 4) if is_3d else (1, 3, 4)
        pad_args = (1, 2, 3, 0, 5, 6) if is_3d else (0, 0, 3, 0, 5, 6)
        crop_args = (3, 2, 8, 0, 1, 4) if is_3d else (0, 0, 8, 0, 1, 4)
        transforms = (
            torchio.CropOrPad(cp_args),
            torchio.ToCanonical(),
            torchio.Resample((1, 1.1, 1.25)),
            torchio.RandomFlip(axes=flip_axes, flip_probability=1),
            torchio.RandomMotion(),
            torchio.RandomGhosting(axes=(0, 1, 2)),
            torchio.RandomSpike(),
            torchio.RandomNoise(),
            torchio.RandomBlur(),
            torchio.RandomSwap(patch_size=swap_patch, num_iterations=5),
            torchio.Lambda(lambda x: 2 * x, types_to_apply=torchio.INTENSITY),
            torchio.RandomBiasField(),
            torchio.RescaleIntensity((0, 1)),
            torchio.ZNormalization(),
            torchio.HistogramStandardization(landmarks_dict),
            elastic,
            torchio.RandomAffine(),
            torchio.OneOf({
                torchio.RandomAffine(): 3,
                elastic: 1,
            }),
            torchio.Pad(pad_args, padding_mode=3),
            torchio.Crop(crop_args),
        )
        return torchio.Compose(transforms)

    def test_transforms_tensor(self):
        tensor = torch.rand(2, 4, 5, 8)
        transform = self.get_transform(channels=('channel_0', 'channel_1'))
        transform(tensor)

    def test_transforms_sample_3d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=True)
        transform(self.sample)

    def test_transforms_sample_2d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=False)
        sample = self.make_2d(self.sample)
        transform(sample)
