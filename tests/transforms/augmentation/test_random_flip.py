import torchio
from ...utils import TorchioTestCase


class TestRandomFlip(TorchioTestCase):
    """Tests for `RandomFlip`."""
    def test_2d(self):
        sample = self.make_2d(self.sample)
        transform = torchio.transforms.RandomFlip(
            axes=(0, 1), flip_probability=1)
        transform(sample)

    def test_wrong_axes(self):
        sample = self.make_2d(self.sample)
        transform = torchio.transforms.RandomFlip(axes=2, flip_probability=1)
        with self.assertRaises(RuntimeError):
            transform(sample)
