#!/usr/bin/env python

"""Tests for Subject."""

import tempfile
from torchio import Subject, ScalarImage, RandomFlip
from ..utils import TorchioTestCase


class TestSubject(TorchioTestCase):
    """Tests for `Subject`."""
    def test_positional_args(self):
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile() as f:
                Subject(ScalarImage(f.name))

    def test_input_dict(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': ScalarImage(f.name)}
            Subject(input_dict)
            Subject(**input_dict)

    def test_no_sample(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': ScalarImage(f.name)}
            subject = Subject(input_dict)
            with self.assertRaises(RuntimeError):
                RandomFlip()(subject)

    def test_history(self):
        transformed = RandomFlip()(self.sample)
        self.assertIs(len(transformed.history), 1)
