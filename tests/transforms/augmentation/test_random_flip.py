import torch
from torchio import RandomFlip
from ...utils import TorchioTestCase


class TestRandomFlip(TorchioTestCase):
    """Tests for `RandomFlip`."""
    def test_2d(self):
        subject = self.make_2d(self.sample_subject)
        transform = RandomFlip(axes=(1, 2), flip_probability=1)
        transformed = transform(subject)
        self.assertTensorEqual(
            subject.t1.data.numpy()[..., ::-1, ::-1],
            transformed.t1.data.numpy(),
        )

    def test_out_of_range_axis(self):
        with self.assertRaises(ValueError):
            RandomFlip(axes=3)

    def test_out_of_range_axis_in_tuple(self):
        with self.assertRaises(ValueError):
            RandomFlip(axes=(0, -1, 2))

    def test_wrong_axes_type(self):
        with self.assertRaises(ValueError):
            RandomFlip(axes=None)

    def test_wrong_flip_probability_type(self):
        with self.assertRaises(ValueError):
            RandomFlip(flip_probability='wrong')

    def test_anatomical_axis(self):
        transform = RandomFlip(axes=['i'], flip_probability=1)
        tensor = torch.rand(1, 2, 3, 4)
        transformed = transform(tensor)
        self.assertTensorEqual(
            tensor.numpy()[..., ::-1],
            transformed.numpy(),
        )
