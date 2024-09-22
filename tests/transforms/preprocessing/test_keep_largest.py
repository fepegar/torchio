import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestKeepLargestComponent(TorchioTestCase):
    """Tests for `KeepLargestComponent`."""

    def test_one_hot(self):
        tensor = torch.as_tensor([1, 0, 1, 1, 0, 1]).reshape(1, 1, 1, 6)
        label_map = tio.LabelMap(tensor=tensor)
        largest = tio.KeepLargestComponent()(label_map)
        assert largest.data.sum() == 2

    def test_multichannel(self):
        label_map = tio.LabelMap(tensor=torch.rand(2, 3, 3, 3) > 1)
        with pytest.raises(RuntimeError):
            tio.KeepLargestComponent()(label_map)
