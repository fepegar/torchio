from torchio import RandomBlur
from ...utils import TorchioTestCase
from numpy.testing import assert_array_equal


class TestRandomBlur(TorchioTestCase):
    """Tests for `RandomBlur`."""
    def test_no_blurring(self):
        transform = RandomBlur(std=0)
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_with_blurring(self):
        transform = RandomBlur(std=(1, 3))
        transformed = transform(self.sample)
        with self.assertRaises(AssertionError):
            assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_negative_std(self):
        with self.assertRaises(ValueError):
            RandomBlur(std=-2)

    def test_std_range_with_negative_min(self):
        with self.assertRaises(ValueError):
            RandomBlur(std=(-0.5, 4))

    def test_wrong_std_type(self):
        with self.assertRaises(ValueError):
            RandomBlur(std='wrong')
