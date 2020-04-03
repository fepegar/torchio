import torchio
from ...utils import TorchioTestCase


class TestRandomMotion(TorchioTestCase):
    """Tests for `RandomMotion`."""
    def test_random_motion(self):
        transform = torchio.transforms.RandomMotion(
            seed=42,
        )
        transformed = transform(self.sample)
        self.sample['t2'][torchio.DATA] = self.sample['t2'][torchio.DATA] - 0.5
        with self.assertWarns(UserWarning):
            transformed = transform(self.sample)
