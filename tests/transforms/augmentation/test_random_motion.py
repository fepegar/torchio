import torchio
from ...utils import TorchioTestCase


class TestRandomMotion(TorchioTestCase):
    """Tests for `RandomMotion`."""
    def test_random_motion(self):
        with self.assertRaises(ValueError):
            transform = torchio.transforms.RandomMotion(num_transforms=0)
