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
            Image('nopath', INTENSITY)

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            Image(5, INTENSITY)

    def test_incompatible_arguments(self):
        with self.assertRaises(ValueError):
            Image(5, INTENSITY, affine=1)

    def test_tensor_flip(self):
        sample_input = torch.ones((4, 30, 30, 30))
        RandomFlip()(sample_input)

    def test_tensor_affine(self):
        sample_input = torch.ones((4, 10, 10, 10))
        RandomAffine()(sample_input)