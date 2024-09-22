import pytest

from torchio.data import PatchSampler

from ...utils import TorchioTestCase


class TestPatchSampler(TorchioTestCase):
    """Tests for `PatchSampler` class."""

    def test_bad_patch_size(self):
        with pytest.raises(ValueError):
            PatchSampler(0)
        with pytest.raises(ValueError):
            PatchSampler(-1)
        with pytest.raises(ValueError):
            PatchSampler(1.5)

    def test_extract_patch(self):
        PatchSampler(1).extract_patch(self.sample_subject, (3, 4, 5))
