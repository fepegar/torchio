from torchio.transforms import ZNormalization
from ...utils import TorchioTestCase


class TestZNormalization(TorchioTestCase):
    """Tests for :py:class:`ZNormalization` class."""

    def test_z_normalization(self):
        transform = ZNormalization()
        transformed = transform(self.sample)
        self.assertAlmostEqual(float(transformed.t1.data.mean()), 0., places=6)
        self.assertAlmostEqual(float(transformed.t1.data.std()), 1.)
