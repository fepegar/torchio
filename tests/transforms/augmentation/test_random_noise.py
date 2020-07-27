from torchio import RandomNoise
from ...utils import TorchioTestCase
from numpy.testing import assert_array_equal


class TestRandomNoise(TorchioTestCase):
    """Tests for `RandomNoise`."""
    def test_no_noise(self):
        transform = RandomNoise(mean=0., std=0.)
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_with_noise(self):
        transform = RandomNoise()
        transformed = transform(self.sample)
        with self.assertRaises(AssertionError):
            assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_constant_noise(self):
        transform = RandomNoise(mean=(5., 5.), std=0.)
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data + 5, transformed.t1.data)

    def test_negative_std(self):
        with self.assertRaises(ValueError):
            RandomNoise(std=-2)

    def test_std_range_with_negative_min(self):
        with self.assertRaises(ValueError):
            RandomNoise(std=(-0.5, 4))

    def test_wrong_std_type(self):
        with self.assertRaises(ValueError):
            RandomNoise(std='wrong')

    def test_wrong_mean_type(self):
        with self.assertRaises(ValueError):
            RandomNoise(mean='wrong')
