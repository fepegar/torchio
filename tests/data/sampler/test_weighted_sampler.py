import torch

import torchio as tio
from torchio.data import WeightedSampler

from ...utils import TorchioTestCase


class TestWeightedSampler(TorchioTestCase):
    """Tests for `WeightedSampler` class."""

    def test_weighted_sampler(self):
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        patch = tio.utils.get_first_item(sampler(subject))
        assert tuple(patch[tio.LOCATION][:3]) == (1, 1, 1)

    def get_sample(self, image_shape):
        t1 = torch.rand(*image_shape)
        prob = torch.zeros_like(t1)
        prob[0, 3, 3, 3] = 1
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=t1),
            prob=tio.ScalarImage(tensor=prob),
        )
        subject = tio.SubjectsDataset([subject])[0]
        return subject

    def test_inconsistent_shape(self):
        # https://github.com/TorchIO-project/torchio/issues/234#issuecomment-675029767
        subject = tio.Subject(
            im1=tio.ScalarImage(tensor=torch.rand(1, 4, 5, 6)),
            im2=tio.ScalarImage(tensor=torch.rand(2, 4, 5, 6)),
        )
        patch_size = 2
        sampler = tio.data.WeightedSampler(patch_size, 'im1')
        next(sampler(subject))
