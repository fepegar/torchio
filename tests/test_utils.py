#!/usr/bin/env python

"""Tests for `utils` package."""

import unittest
import torch
import numpy as np
from torchio import LABEL, INTENSITY
from torchio.utils import check_consistent_shape


class TestUtils(unittest.TestCase):
    """Tests for `utils` module."""

    def get_sample(self, consistent):
        shape = 1, 10, 20, 30
        affine = np.diag((1, 2, 3, 1))
        affine[:3, 3] = 40, 50, 60
        shape2 = 1, 20, 10, 30
        sample = {
            't1': dict(
                data=self.getRandomData(shape),
                affine=affine,
                type=INTENSITY,
            ),
            't2': dict(
                data=self.getRandomData(shape if consistent else shape2),
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

    def test_sample_shapes(self):
        good_sample = self.get_sample(consistent=True)
        bad_sample = self.get_sample(consistent=False)
        check_consistent_shape(good_sample)
        with self.assertRaises(ValueError):
            check_consistent_shape(bad_sample)
