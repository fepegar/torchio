import pytest

import torchio as tio

from ..utils import TorchioTestCase


class TestIXI(TorchioTestCase):
    """Tests for `ixi` module."""

    def test_not_downloaded(self):
        with pytest.raises(RuntimeError):
            tio.datasets.IXI('testing123', download=False)

    def test_tiny_not_downloaded(self):
        with pytest.raises(RuntimeError):
            tio.datasets.IXITiny('testing123', download=False)
