import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestLabelSampler(TorchioTestCase):
    """Tests for `LabelSampler` class."""

    def test_label_sampler(self):
        sampler = tio.LabelSampler(5)
        for patch in sampler(self.sample_subject, num_patches=10):
            patch_center = patch['label'][tio.DATA][0, 2, 2, 2]
            assert patch_center == 1

    def test_label_probabilities(self):
        labels = torch.Tensor((0, 0, 1, 1, 2, 1, 0)).reshape(1, 1, 1, -1)
        subject = tio.Subject(
            label=tio.Image(tensor=labels, type=tio.LABEL),
        )
        subject = tio.SubjectsDataset([subject])[0]
        probs_dict = {0: 0, 1: 50, 2: 25, 3: 25}
        patch_size = (1, 1, 5)
        sampler = tio.LabelSampler(patch_size, label_probabilities=probs_dict)
        probabilities = sampler.get_probability_map(subject)
        fixture = torch.Tensor((0, 0, 1 / 4, 1 / 4, 1 / 4, 0, 0))
        assert torch.all(probabilities.squeeze().eq(fixture))

    def test_inconsistent_shape(self):
        # https://github.com/TorchIO-project/torchio/issues/234#issuecomment-675029767
        subject = tio.Subject(
            im1=tio.ScalarImage(tensor=torch.rand(2, 4, 5, 6)),
            im2=tio.LabelMap(tensor=torch.rand(1, 4, 5, 6)),
        )
        patch_size = 2
        sampler = tio.LabelSampler(patch_size, 'im2')
        next(sampler(subject))

    def test_multichannel_label_sampler(self):
        subject = tio.Subject(
            label=tio.LabelMap(
                tensor=torch.tensor(
                    [
                        [[[1, 1]]],
                        [[[0, 1]]],
                    ],
                ),
            ),
        )
        patch_size = 1
        sampler = tio.LabelSampler(
            patch_size,
            'label',
            label_probabilities={0: 1, 1: 1},
        )
        # There are 2 voxels in the image, channels have same probabilities,
        # 1st voxel has probability 0.5 * 0.5 + 0 * 0.5 of being chosen while
        # 2nd voxel has probability 0.5 * 0.5 + 1 * 0.5 of being chosen.
        probabilities = sampler.get_probability_map(subject)
        fixture = torch.Tensor((1 / 4, 3 / 4))
        assert torch.all(probabilities.squeeze().eq(fixture))

    def test_no_labelmap(self):
        im = tio.ScalarImage(tensor=torch.rand(1, 1, 1, 1))
        subject = tio.Subject(image=im, no_label=im)
        sampler = tio.LabelSampler(1)
        with pytest.raises(RuntimeError):
            next(sampler(subject))

    def test_empty_map(self):
        # https://github.com/TorchIO-project/torchio/issues/392
        im = tio.ScalarImage(tensor=torch.rand(1, 6, 6, 6))
        label = torch.zeros(1, 6, 6, 6)
        label[..., 0] = 1  # voxels far from center
        label_im = tio.LabelMap(tensor=label)
        subject = tio.Subject(image=im, label=label_im)
        sampler = tio.LabelSampler(4)
        with pytest.raises(RuntimeError):
            next(sampler(subject))
