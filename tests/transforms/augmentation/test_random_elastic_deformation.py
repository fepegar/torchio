#!/usr/bin/env python

import unittest
import numpy as np
import torchio
from torchio import INTENSITY, LABEL, DATA
from torchio.transforms import RandomElasticDeformation
from ...utils import TorchioTestCase


class TestRandomElasticDeformation(TorchioTestCase):
    """Tests for `RandomElasticDeformation`."""

    def test_random_elastic_deformation(self):
        transform = RandomElasticDeformation(
            proportion_to_augment=1,
            seed=42,
        )
        keys = ('t1', 't2', 'label')
        fixtures = 2328.8125, 2317.3125, 2308
        transformed = transform(self.sample)
        for key, fixture in zip(keys, fixtures):
            data = transformed[key][DATA]
            total = data.sum().item()
            self.assertAlmostEqual(total, fixture)

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
