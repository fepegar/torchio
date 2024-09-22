import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomSampler(TorchioTestCase):
    def test_not_implemented(self):
        sampler = tio.data.sampler.RandomSampler(1)
        with pytest.raises(NotImplementedError):
            sampler(self.sample_subject, 5)
        with pytest.raises(NotImplementedError):
            sampler.get_probability_map(self.sample_subject)
