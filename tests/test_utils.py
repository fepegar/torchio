#!/usr/bin/env python

"""Tests for `utils` package."""

import unittest
import torch
import numpy as np
import SimpleITK as sitk
from torchio import RandomFlip
from torchio.utils import (
    to_tuple,
    get_stem,
    guess_type,
    nib_to_sitk,
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


class TestNibabelToSimpleITK(TorchioTestCase):
    def setUp(self):
        super().setUp()
        self.affine = np.eye(4)

    def test_wrong_dims(self):
        with self.assertRaises(ValueError):
            nib_to_sitk(np.random.rand(10, 10), self.affine)

    def test_2d_single(self):
        data = np.random.rand(1, 1, 10, 12)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 2
        assert image.GetSize() == (10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_2d_multi(self):
        data = np.random.rand(5, 1, 10, 12)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 2
        assert image.GetSize() == (10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5

    def test_2d_3d_single(self):
        data = np.random.rand(1, 1, 10, 12)
        image = nib_to_sitk(data, self.affine, force_3d=True)
        assert image.GetDimension() == 3
        assert image.GetSize() == (1, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_2d_3d_multi(self):
        data = np.random.rand(5, 1, 10, 12)
        image = nib_to_sitk(data, self.affine, force_3d=True)
        assert image.GetDimension() == 3
        assert image.GetSize() == (1, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5

    def test_3d_single(self):
        data = np.random.rand(1, 8, 10, 12)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 3
        assert image.GetSize() == (8, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_3d_multi(self):
        data = np.random.rand(5, 8, 10, 12)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 3
        assert image.GetSize() == (8, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5
