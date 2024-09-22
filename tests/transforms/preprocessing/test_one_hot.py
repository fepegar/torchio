import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestOneHot(TorchioTestCase):
    """Tests for `OneHot`."""

    def test_one_hot(self):
        image = self.sample_subject.label
        one_hot = tio.OneHot(num_classes=3)(image)
        assert one_hot.num_channels == 3

    def test_multichannel(self):
        label_map = tio.LabelMap(tensor=torch.rand(2, 3, 3, 3) > 1)
        with pytest.raises(RuntimeError):
            tio.OneHot()(label_map)

    def test_inverse(self):
        one_hot = tio.OneHot()
        subject_one_hot = one_hot(self.sample_subject)
        subject_back = subject_one_hot.apply_inverse_transform()
        self.assert_tensor_equal(
            self.sample_subject.label.data,
            subject_back.label.data,
        )
