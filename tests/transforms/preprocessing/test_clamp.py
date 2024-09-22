import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestClamp(TorchioTestCase):
    """Tests for :class:`tio.Clamp` class."""

    def test_out_min_max(self):
        transform = tio.Clamp(out_min=0, out_max=1)
        transformed = transform(self.sample_subject)
        assert transformed.t1.data.min() == 0
        assert transformed.t1.data.max() == 1

    def test_ct(self):
        ct_max = 1500
        ct_min = -2000
        ct_range = ct_max - ct_min
        tensor = torch.rand(1, 30, 30, 30) * ct_range + ct_min
        ct = tio.ScalarImage(tensor=tensor)
        ct_air = -1000
        ct_bone = 1000
        clamp = tio.Clamp(ct_air, ct_bone)
        clamped = clamp(ct)
        assert clamped.data.min() == ct_air
        assert clamped.data.max() == ct_bone

    def test_too_many_values_for_out_min(self):
        with pytest.raises(TypeError):
            clamp = tio.Clamp(out_min=(1, 2))
            clamp(self.sample_subject)

    def test_too_many_values_for_out_max(self):
        with pytest.raises(TypeError):
            clamp = tio.Clamp(out_max=(1, 2))
            clamp(self.sample_subject)

    def test_wrong_out_min_type(self):
        with pytest.raises(TypeError):
            clamp = tio.Clamp(out_min='foo')
            clamp(self.sample_subject)

    def test_wrong_out_max_type(self):
        with pytest.raises(TypeError):
            clamp = tio.Clamp(out_max='foo')
            clamp(self.sample_subject)
