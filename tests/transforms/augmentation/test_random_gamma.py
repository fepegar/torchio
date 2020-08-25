from torchio import RandomGamma
from ...utils import TorchioTestCase
from numpy.testing import assert_array_equal


class TestRandomGamma(TorchioTestCase):
    """Tests for `RandomGamma`."""
    def test_with_zero_gamma(self):
        transform = RandomGamma(log_gamma_range=0)
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_with_non_zero_gamma(self):
        transform = RandomGamma(log_gamma_range=(0.1, 0.3))
        transformed = transform(self.sample)
        with self.assertRaises(AssertionError):
            assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_with_high_gamma(self):
        transform = RandomGamma(log_gamma_range=(100, 100))
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data == 1, transformed.t1.data)

    def test_with_low_gamma(self):
        transform = RandomGamma(log_gamma_range=(-100, -100))
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data > 0, transformed.t1.data)

    def test_wrong_gamma_type(self):
        with self.assertRaises(ValueError):
            RandomGamma(log_gamma_range='wrong')
