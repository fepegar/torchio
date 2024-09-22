import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestCrop(TorchioTestCase):
    def test_tensor_single_channel(self):
        crop = tio.Crop(1)
        assert crop(torch.rand(1, 10, 10, 10)).shape == (1, 8, 8, 8)

    def test_tensor_multi_channel(self):
        crop = tio.Crop(1)
        assert crop(torch.rand(3, 10, 10, 10)).shape == (3, 8, 8, 8)
