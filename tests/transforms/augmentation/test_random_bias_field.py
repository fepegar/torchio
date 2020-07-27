from torchio import RandomBiasField
from ...utils import TorchioTestCase
from numpy.testing import assert_array_equal


class TestRandomBiasField(TorchioTestCase):
    """Tests for `RandomBiasField`."""
    def test_no_bias(self):
        transform = RandomBiasField(coefficients=0.)
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_with_bias(self):
        transform = RandomBiasField(coefficients=0.1)
        transformed = transform(self.sample)
        with self.assertRaises(AssertionError):
            assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_wrong_coefficient_type(self):
        with self.assertRaises(ValueError):
            RandomBiasField(coefficients='wrong')

    def test_negative_order(self):
        with self.assertRaises(ValueError):
            RandomBiasField(order=-1)

    def test_wrong_order_type(self):
        with self.assertRaises(TypeError):
            RandomBiasField(order='wrong')
