from torchio.data.sampler import RandomSampler
from ...utils import TorchioTestCase


class TestRandomSampler(TorchioTestCase):
    """Tests for `RandomSampler` class."""
    def test_not_implemented(self):
        sampler = RandomSampler(1)
        with self.assertRaises(NotImplementedError):
            sampler(self.sample_subject, 5)
        with self.assertRaises(NotImplementedError):
            sampler.get_probability_map(self.sample_subject)
