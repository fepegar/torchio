import numpy as np
from torchio.transforms import RandomDownsample
from ...utils import TorchioTestCase


class TestRandomDownsample(TorchioTestCase):
    """Tests for `RandomDownsample`."""

    def test_downsample(self):
        transform = RandomDownsample(
            axes=1,
            downsampling=(2., 2.)
        )
        transformed = transform(self.sample)
        self.assertEqual(self.sample.spacing[1] * 2, transformed.spacing[1])

    def test_out_of_range_axis(self):
        with self.assertRaises(ValueError):
            RandomDownsample(axes=3)

    def test_out_of_range_axis_in_tuple(self):
        with self.assertRaises(ValueError):
            RandomDownsample(axes=(0, -1, 2))

    def test_wrong_axes_type(self):
        with self.assertRaises(ValueError):
            RandomDownsample(axes='wrong')

    def test_wrong_downsampling_type(self):
        with self.assertRaises(ValueError):
            RandomDownsample(downsampling='wrong')

    def test_below_one_downsampling(self):
        with self.assertRaises(ValueError):
            RandomDownsample(downsampling=0.2)
