import torch
import torchio as tio
from ...utils import TorchioTestCase


class TestZNormalization(TorchioTestCase):
    """Tests for :class:`ZNormalization` class."""

    def test_z_normalization(self):
        transform = tio.ZNormalization()
        transformed = transform(self.sample_subject)
        self.assertAlmostEqual(float(transformed.t1.data.mean()), 0., places=6)
        self.assertAlmostEqual(float(transformed.t1.data.std()), 1.)

    def test_no_std(self):
        image = tio.ScalarImage(tensor=torch.ones(1, 2, 2, 2))
        with self.assertRaises(RuntimeError):
            tio.ZNormalization()(image)
