import tempfile
import unittest
from pathlib import Path
from torchio.datasets import IXI, IXITiny


class TestIXI(unittest.TestCase):
    """Tests for `ixi` module."""

    def test_ixi(self):
        root_dir = Path(tempfile.gettempdir(), 'ixi_tiny')
        dataset = IXITiny(root_dir, download=True)

    def test_not_downloaded(self):
        with self.assertRaises(RuntimeError):
            dataset = IXI('testing123', download=False)

    def test_tiny_not_downloaded(self):
        with self.assertRaises(RuntimeError):
            dataset = IXITiny('testing123', download=False)
