from torchio.datasets import IXI, IXITiny
from ..utils import TorchioTestCase


class TestIXI(TorchioTestCase):
    """Tests for `ixi` module."""

    def test_ixi(self):
        self.get_ixi_tiny()

    def test_not_downloaded(self):
        with self.assertRaises(RuntimeError):
            dataset = IXI('testing123', download=False)

    def test_tiny_not_downloaded(self):
        with self.assertRaises(RuntimeError):
            dataset = IXITiny('testing123', download=False)
