import torch
from torchio import ScalarImage, RandomAnisotropy
from ...utils import TorchioTestCase


class TestRandomAnisotropy(TorchioTestCase):
    """Tests for `RandomAnisotropy`."""

    def test_downsample(self):
        transform = RandomAnisotropy(
            axes=1,
            downsampling=(2., 2.)
        )
        transformed = transform(self.sample_subject)
        self.assertEqual(
            self.sample_subject.spacing[1],
            transformed.spacing[1],
        )

    def test_out_of_range_axis(self):
        with self.assertRaises(ValueError):
            RandomAnisotropy(axes=3)

    def test_out_of_range_axis_in_tuple(self):
        with self.assertRaises(ValueError):
            RandomAnisotropy(axes=(0, -1, 2))

    def test_wrong_axes_type(self):
        with self.assertRaises(ValueError):
            RandomAnisotropy(axes='wrong')

    def test_wrong_downsampling_type(self):
        with self.assertRaises(ValueError):
            RandomAnisotropy(downsampling='wrong')

    def test_below_one_downsampling(self):
        with self.assertRaises(ValueError):
            RandomAnisotropy(downsampling=0.2)

    def test_2d_rgb(self):
        image = ScalarImage(tensor=torch.rand(3, 4, 5, 6))
        RandomAnisotropy()(image)
