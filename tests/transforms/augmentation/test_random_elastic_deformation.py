#!/usr/bin/env python

import unittest
import numpy as np
import torchio
from torchio import INTENSITY, LABEL, DATA
from torchio.transforms import RandomElasticDeformation


class TestRandomElasticDeformation(unittest.TestCase):
    """Tests for `RandomElasticDeformation`."""

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
            ),
            't2': dict(
                data=self.getRandomData(shape),
                affine=affine,
                type=INTENSITY,
            ),
            'label': dict(
                data=(self.getRandomData(shape) > 0.5).astype(np.float32),
                affine=affine,
                type=LABEL,
            ),
        }

    @staticmethod
    def getRandomData(shape):
        return np.random.rand(*shape)

    def test_random_elastic_deformation(self):
        transform = RandomElasticDeformation(
            proportion_to_augment=1,
            seed=42,
        )
        keys = ('t1', 't2', 'label')
        fixtures = 2463.8931905687296, 2465.493324966148, 2532
        transformed = transform(self.sample)
        for key, fixture in zip(keys, fixtures):
            self.assertAlmostEqual(transformed[key][DATA].sum(), fixture)

    def test_random_elastic_deformation_inputs_pta(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(proportion_to_augment=1.5)
        with self.assertRaises(ValueError):
            RandomElasticDeformation(proportion_to_augment=-1)

    def test_random_elastic_deformation_inputs_interpolation(self):
        with self.assertRaises(TypeError):
            RandomElasticDeformation(image_interpolation=1)
        with self.assertRaises(TypeError):
            RandomElasticDeformation(image_interpolation='linear')
