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
        with self.assertRaises(RuntimeError):
            tio.OneHot()(label_map)
