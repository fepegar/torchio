import torch
import torchio
import numpy as np
from torchio.data import UniformSampler
from ...utils import TorchioTestCase


class TestUniformSampler(TorchioTestCase):
    """Tests for `UniformSampler` class."""

    def test_uniform_probabilities(self):
        sampler = UniformSampler(5)
        probabilities = sampler.get_probability_map(self.sample)
        fixtures = torch.ones_like(probabilities)
        assert torch.all(probabilities.eq(fixtures))

    def test_processed_uniform_probabilities(self):
        sampler = UniformSampler(5)
        probabilities = sampler.get_probability_map(self.sample)
        probabilities = sampler.process_probability_map(probabilities)
        fixtures = np.zeros_like(probabilities)
        # Other positions cannot be patch centers
        fixtures[2:-2, 2:-2, 2:-2] = probabilities[2, 2, 2]
        self.assertAlmostEqual(probabilities.sum(), 1)
        assert np.equal(probabilities, fixtures).all()

    def test_incosistent_shape(self):
        # https://github.com/fepegar/torchio/issues/234#issuecomment-675029767
        sample = torchio.Subject(
            im1=torchio.ScalarImage(tensor=torch.rand(1, 4, 5, 6)),
            im2=torchio.ScalarImage(tensor=torch.rand(2, 4, 5, 6)),
        )
        patch_size = 2
        sampler = UniformSampler(patch_size)
        next(sampler(sample))
