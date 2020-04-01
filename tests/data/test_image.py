#!/usr/bin/env python

"""Tests for Image."""

import torchio
from torchio import INTENSITY, Image
from ..utils import TorchioTestCase


class TestImage(TorchioTestCase):
    """Tests for `Image`."""

    def test_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Image('t1', 'nopath', INTENSITY)

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            Image('t1', 5, INTENSITY)
