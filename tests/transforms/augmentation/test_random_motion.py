#!/usr/bin/env python

import unittest
import numpy as np
import torchio
from torchio import INTENSITY, LABEL, DATA
from ...utils import TorchioTestCase


class TestRandomMotion(TorchioTestCase):
    """Tests for `RandomMotion`."""
    def test_random_motion(self):
        transform = torchio.transforms.RandomMotion(
            seed=42,
        )
        transformed = transform(self.sample)
        self.sample['t2'][DATA] = self.sample['t2'][DATA] - 0.5
        with self.assertWarns(UserWarning):
            transformed = transform(self.sample)
