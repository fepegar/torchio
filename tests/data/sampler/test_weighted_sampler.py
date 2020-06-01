import torch
import torchio
from torchio.data import WeightedSampler
from ...utils import TorchioTestCase


class TestWeightedSampler(TorchioTestCase):
    """Tests for `WeightedSampler` class."""

    def test_weighted_sampler(self):
        sample = self.get_sample((7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        patch = next(iter(sampler(sample)))
        self.assertEqual(tuple(patch['index_ini']), (1, 1, 1))

    def get_sample(self, image_shape):
        t1 = torch.rand(*image_shape)
        prob = torch.zeros_like(t1)
        prob[3, 3, 3] = 1
        subject = torchio.Subject(
            t1=torchio.Image(tensor=t1),
            prob=torchio.Image(tensor=prob),
        )
        sample = torchio.ImagesDataset([subject])[0]
        return sample
