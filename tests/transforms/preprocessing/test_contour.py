import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestContour(TorchioTestCase):
    """Tests for `Contour`."""

    def test_one_hot(self):
        image = self.sample_subject.label
        tio.Contour()(image)

    def test_multichannel(self):
        label_map = tio.LabelMap(tensor=torch.rand(2, 3, 3, 3) > 1)
        with pytest.raises(RuntimeError):
            tio.Contour()(label_map)
