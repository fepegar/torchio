#!/usr/bin/env python

"""Tests for Subject."""

import tempfile
import torchio
from torchio import INTENSITY, Subject, Image
from ..utils import TorchioTestCase

class TestSubject(TorchioTestCase):
    """Tests for `Subject`."""
    def test_duplicate_image_name(self):
        with self.assertRaises(KeyError):
            with tempfile.NamedTemporaryFile() as f:
                Subject(
                    Image('t1', f.name, INTENSITY),
                    Image('t1', f.name, INTENSITY),
                )
            self.iterate_dataset([images])
