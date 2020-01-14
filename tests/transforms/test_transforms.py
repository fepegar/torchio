#!/usr/bin/env python

import unittest
import torch
import numpy as np
from torchio import INTENSITY, LABEL

from torchio.transforms import (
    RandomFlip,
    RandomNoise,
    RandomBiasField,
    RandomElasticDeformation,
    RandomAffine,
    RandomMotion,
    Rescale,
    ZNormalization,
    HistogramStandardization,
    Pad,
    Crop,
)


class TestTransforms(unittest.TestCase):
    """Tests for all transforms."""

    def get_sample(self):
        shape = 1, 10, 20, 30
        np.random.seed(42)
        affine = np.diag((1, 2, 3, 1))
        affine[:3, 3] = 40, 50, 60
        sample = {
            't1': dict(
                data=self.getRandomData(shape),
                affine=affine,
                type=INTENSITY,
            ),
            't2': dict(
                data=self.getRandomData(shape),
                affine=affine,
                type=INTENSITY,
            ),
            'label': dict(
                data=(self.getRandomData(shape) > 0.5).float(),
                affine=affine,
                type=LABEL,
            ),
        }
        return sample

    @staticmethod
    def getRandomData(shape):
        return torch.rand(*shape)

    def test_transforms(self):
        landmarks_dict = dict(
            t1=np.linspace(0, 100, 13),
            t2=np.linspace(0, 100, 13),
        )
        random_transforms = (
            RandomFlip(axes=(0, 1, 2), flip_probability=1),
            RandomNoise(),
            RandomBiasField(),
            RandomElasticDeformation(proportion_to_augment=1),
            RandomAffine(),
            RandomMotion(proportion_to_augment=1),
        )
        preprocessing_transforms = (
            Rescale(),
            ZNormalization(),
            HistogramStandardization(landmarks_dict=landmarks_dict),
            Pad((1, 2, 3, 0, 5, 6)),
            Crop((3, 2, 8, 0, 1, 4)),
        )
        for transform in random_transforms:
            sample = self.get_sample()
            transformed = transform(sample)

        for transform in preprocessing_transforms:
            sample = self.get_sample()
            transformed = transform(sample)
