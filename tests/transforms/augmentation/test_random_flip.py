from torchio import RandomFlip
from numpy.testing import assert_array_equal
from ...utils import TorchioTestCase


class TestRandomFlip(TorchioTestCase):
    """Tests for `RandomFlip`."""
    def test_2d(self):
        sample = self.make_2d(self.sample)
        transform = RandomFlip(axes=(1, 2), flip_probability=1)
        transformed = transform(sample)
        assert_array_equal(
            sample.t1.data.numpy()[:, :, ::-1, ::-1],
            transformed.t1.data.numpy())

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
