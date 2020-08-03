import torch
from torchio.transforms import Crop
from ...utils import TorchioTestCase


class TestCrop(TorchioTestCase):
    """Tests for `Crop`."""
    def test_tensor_single_channel(self):
        crop = Crop(1)
        self.assertEqual(crop(torch.rand(1, 10, 10, 10)).shape, (1, 8, 8, 8))

    def test_tensor_multi_channel(self):
        crop = Crop(1)
        self.assertEqual(crop(torch.rand(3, 10, 10, 10)).shape, (3, 8, 8, 8))
