#!/usr/bin/env python

"""Tests for `utils` package."""

import unittest
import torch
import numpy as np
from torchio import LABEL, INTENSITY
from torchio.utils import (
    to_tuple,
    get_stem,
    guess_type,
)
from .utils import TorchioTestCase


class TestUtils(TorchioTestCase):
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

    def test_to_tuple(self):
        assert to_tuple(1) == (1,)
        assert to_tuple((1,)) == (1,)
        assert to_tuple(1, length=3) == (1, 1, 1)
        assert to_tuple((1, 2)) == (1, 2)
        assert to_tuple((1, 2), length=3) == (1, 2)
        assert to_tuple([1, 2], length=3) == (1, 2)

    def test_get_stem(self):
        assert get_stem('/home/image.nii.gz') == 'image'
        assert get_stem('/home/image.nii') == 'image'
        assert get_stem('/home/image.nrrd') == 'image'

    def test_guess_type(self):
        assert guess_type('None') is None
        assert isinstance(guess_type('1'), int)
        assert isinstance(guess_type('1.5'), float)
        assert isinstance(guess_type('(1, 3, 5)'), tuple)
        assert isinstance(guess_type('(1,3,5)'), tuple)
        assert isinstance(guess_type('[1,3,5]'), list)
        assert isinstance(guess_type('test'), str)

    def test_check_consistent_shape(self):
        good_sample = self.sample
        bad_sample = self.get_inconsistent_sample()
        good_sample.check_consistent_shape()
        with self.assertRaises(ValueError):
            bad_sample.check_consistent_shape()
