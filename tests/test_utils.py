#!/usr/bin/env python

"""Tests for `utils` package."""

import unittest
import torch
import numpy as np
import SimpleITK as sitk
from torchio import LABEL, INTENSITY, RandomFlip
from torchio.utils import (
    to_tuple,
    get_stem,
    guess_type,
    sitk_to_nib,
    apply_transform_to_file,
)
from .utils import TorchioTestCase


class TestUtils(TorchioTestCase):
    """Tests for `utils` module."""

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

    def test_apply_transform_to_file(self):
        transform = RandomFlip()
        apply_transform_to_file(
            self.get_image_path('input'),
            transform,
            self.get_image_path('output'),
            verbose=True,
        )

    def test_sitk_to_nib(self):
        data = np.random.rand(10, 10)
        image = sitk.GetImageFromArray(data)
        tensor, affine = sitk_to_nib(image)
        self.assertAlmostEqual(data.sum(), tensor.sum())
