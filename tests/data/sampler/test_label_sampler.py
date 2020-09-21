import torch
import torchio
from torchio.data import LabelSampler
from ...utils import TorchioTestCase


class TestLabelSampler(TorchioTestCase):
    """Tests for `LabelSampler` class."""

    def test_label_sampler(self):
        sampler = LabelSampler(5)
        for patch in sampler(self.sample, num_patches=10):
            patch_center = patch['label'][torchio.DATA][0, 2, 2, 2]
            self.assertEqual(patch_center, 1)

    def test_label_probabilities(self):
        labels = torch.Tensor((0, 0, 1, 1, 2, 1, 0)).reshape(1, 1, 1, -1)
        subject = torchio.Subject(
            label=torchio.Image(tensor=labels, type=torchio.LABEL),
        )
        sample = torchio.SubjectsDataset([subject])[0]
        probs_dict = {0: 0, 1: 50, 2: 25, 3: 25}
        sampler = LabelSampler(5, 'label', label_probabilities=probs_dict)
        probabilities = sampler.get_probability_map(sample)
        fixture = torch.Tensor((0, 0, 2 / 12, 2 / 12, 3 / 12, 2 / 12, 0))
        assert torch.all(probabilities.squeeze().eq(fixture))

    def test_inconsistent_shape(self):
        # https://github.com/fepegar/torchio/issues/234#issuecomment-675029767
        sample = torchio.Subject(
            im1=torchio.ScalarImage(tensor=torch.rand(2, 4, 5, 6)),
            im2=torchio.LabelMap(tensor=torch.rand(1, 4, 5, 6)),
        )
        patch_size = 2
        sampler = LabelSampler(patch_size, 'im2')
        next(sampler(sample))

    def test_multichannel_label_sampler(self):
        sample = torchio.Subject(
            label=torchio.LabelMap(
                tensor=torch.tensor(
                    [
                        [[[1, 1]]],
                        [[[0, 1]]]
                    ]
                )
            )
        )
        patch_size = 1
        sampler = LabelSampler(
            patch_size,
            'label',
            label_probabilities={0: 1, 1: 1}
        )
        # There are 2 voxels in the image, channels have same probabilities,
        # 1st voxel has probability 0.5 * 0.5 + 0 * 0.5 of being chosen while
        # 2nd voxel has probability 0.5 * 0.5 + 1 * 0.5 of being chosen.
        probabilities = sampler.get_probability_map(sample)
        fixture = torch.Tensor((1 / 4, 3 / 4))
        assert torch.all(probabilities.squeeze().eq(fixture))
