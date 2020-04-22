#!/usr/bin/env python

"""Tests for Subject."""

import tempfile
from torchio import INTENSITY, Subject, Image, RandomFlip
from ..utils import TorchioTestCase


class TestSubject(TorchioTestCase):
    """Tests for `Subject`."""
    def test_positional_args(self):
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile() as f:
                Subject(Image(f.name, INTENSITY))

    def test_input_dict(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': Image(f.name, INTENSITY)}
            Subject(input_dict)
            Subject(**input_dict)

    def test_no_sample(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': Image(f.name, INTENSITY)}
            subject = Subject(input_dict)
            with self.assertRaises(RuntimeError):
                RandomFlip()(subject)
