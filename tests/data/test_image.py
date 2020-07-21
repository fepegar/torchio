#!/usr/bin/env python

"""Tests for Image."""

import torch
from torchio import INTENSITY, Image
from ..utils import TorchioTestCase
from torchio import RandomFlip, RandomAffine


class TestImage(TorchioTestCase):
    """Tests for `Image`."""

    def test_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Image('nopath', type=INTENSITY)

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            Image(5, type=INTENSITY)

    def test_wrong_affine(self):
        with self.assertRaises(TypeError):
            Image(5, type=INTENSITY, affine=1)

    def test_tensor_flip(self):
        sample_input = torch.ones((4, 30, 30, 30))
        RandomFlip()(sample_input)

    def test_tensor_affine(self):
        sample_input = torch.ones((4, 10, 10, 10))
        RandomAffine()(sample_input)

    def test_crop_attributes(self):
        cropped = self.sample.crop((1, 1, 1), (5, 5, 5))
        self.assertIs(self.sample.t1['pre_affine'], cropped.t1['pre_affine'])
