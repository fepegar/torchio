import torch
import torchio
import numpy as np
from torchio.data import PatchSampler
from ...utils import TorchioTestCase


class TestPatchSampler(TorchioTestCase):
    """Tests for `PatchSampler` class."""
    def test_bad_patch_size(self):
        with self.assertRaises(ValueError):
            PatchSampler(0)
        with self.assertRaises(ValueError):
            PatchSampler(-1)
        with self.assertRaises(ValueError):
            PatchSampler(1.5)

    def test_extract_patch(self):
        PatchSampler(1).extract_patch(self.sample, (3, 4, 5))
