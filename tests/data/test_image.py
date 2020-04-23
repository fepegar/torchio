#!/usr/bin/env python

"""Tests for Image."""

from torchio import INTENSITY, Image
from ..utils import TorchioTestCase


class TestImage(TorchioTestCase):
    """Tests for `Image`."""

    def test_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Image('nopath', INTENSITY)

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            Image(5, INTENSITY)

    def test_bad_key(self):
        with self.assertRaises(ValueError):
            Image(5, INTENSITY, affine=1)
