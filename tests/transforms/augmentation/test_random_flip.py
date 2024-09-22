import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomFlip(TorchioTestCase):
    def test_2d(self):
        subject = self.make_2d(self.sample_subject)
        transform = tio.RandomFlip(axes=(1, 2), flip_probability=1)
        transformed = transform(subject)
        self.assert_tensor_equal(
            torch.from_numpy(subject.t1.data.numpy()[..., ::-1, ::-1].copy()),
            transformed.t1.data,
        )

    def test_out_of_range_axis(self):
        with pytest.raises(ValueError):
            tio.RandomFlip(axes=3)

    def test_out_of_range_axis_in_tuple(self):
        with pytest.raises(ValueError):
            tio.RandomFlip(axes=(0, -1, 2))

    def test_wrong_axes_type(self):
        with pytest.raises(ValueError):
            tio.RandomFlip(axes=None)

    def test_wrong_flip_probability_type(self):
        with pytest.raises(ValueError):
            tio.RandomFlip(flip_probability='wrong')

    def test_anatomical_axis(self):
        transform = tio.RandomFlip(axes=['i'], flip_probability=1)
        tensor = torch.rand(1, 2, 3, 4)
        transformed = transform(tensor)
        self.assert_tensor_equal(
            torch.from_numpy(tensor.numpy()[..., ::-1].copy()),
            transformed,
        )
