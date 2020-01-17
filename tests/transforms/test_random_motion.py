#!/usr/bin/env python

import unittest
import numpy as np
import torchio
from torchio import INTENSITY, LABEL, DATA


class TestRandomMotion(unittest.TestCase):
    """Tests for `RandomMotion`."""

    def setUp(self):
        """Set up test fixtures, if any."""
        shape = 1, 10, 20, 30
        np.random.seed(42)
        affine = np.diag((1, 2, 3, 1))
        affine[:3, 3] = 40, 50, 60
        self.sample = {
            't1': dict(
                data=self.getRandomData(shape),
                affine=affine,
                type=INTENSITY,
                stem='t1',
            ),
            't2': dict(
                data=self.getRandomData(shape),
                affine=affine,
                type=INTENSITY,
                stem='t2',
            ),
            'label': dict(
                data=(self.getRandomData(shape) > 0.5).astype(np.float32),
                affine=affine,
                type=LABEL,
                stem='label',
            ),
        }

    @staticmethod
    def getRandomData(shape):
        return np.random.rand(*shape)

    def test_random_motion(self):
        transform = torchio.transforms.RandomMotion(
            proportion_to_augment=1,
            seed=42,
        )
        transformed = transform(self.sample)
        transformed['t2'][DATA] = transformed['t2'][DATA] - 0.5
        with self.assertWarns(UserWarning):
            transformed = transform(self.sample)
