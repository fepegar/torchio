#!/usr/bin/env python

import torch
import numpy as np
import torchio
from ..utils import TorchioTestCase


class TestTransforms(TorchioTestCase):
    """Tests for all transforms."""

    def get_transform(self, channels):
        landmarks_dict = {
            channel: np.linspace(0, 100, 13) for channel in channels
        }
        elastic = torchio.RandomElasticDeformation(max_displacement=1)
        transforms = (
            torchio.CropOrPad((9, 21, 30)),
            torchio.ToCanonical(),
            torchio.Resample((1, 1.1, 1.25)),
            torchio.RandomFlip(axes=(0, 1, 2), flip_probability=1),
            torchio.RandomMotion(),
            torchio.RandomGhosting(axes=(0, 1, 2)),
            torchio.RandomSpike(),
            torchio.RandomNoise(),
            torchio.RandomBlur(),
            torchio.RandomSwap(patch_size=2, num_iterations=5),
            torchio.Lambda(lambda x: 2 * x, types_to_apply=torchio.INTENSITY),
            torchio.RandomBiasField(),
            torchio.RescaleIntensity((0, 1)),
            torchio.ZNormalization(),
            torchio.HistogramStandardization(landmarks_dict=landmarks_dict),
            elastic,
            torchio.RandomAffine(),
            torchio.OneOf({torchio.RandomAffine(): 3, elastic: 1}),
            torchio.Pad((1, 2, 3, 0, 5, 6), padding_mode='constant', fill=3),
            torchio.Crop((3, 2, 8, 0, 1, 4)),
        )
        return torchio.Compose(transforms)

    def test_transforms_sample(self):
        transform = self.get_transform(channels=('t1', 't2'))
        transform(self.sample)

    def test_transforms_tensor(self):
        tensor = torch.rand(2, 4, 5, 8)
        transform = self.get_transform(channels=('channel_0', 'channel_1'))
        transform(tensor)
