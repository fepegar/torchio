import tempfile
import unittest
from pathlib import Path
from ..utils import TorchioTestCase
from torchio.data import LabelSampler


class TestLabelSampler(TorchioTestCase):
    """Tests for `LabelSampler` class."""

    def test_label_sampler(self):
        sampler = LabelSampler(self.sample, 5)
        next(iter(sampler))
