from torchio import RandomBiasField
from ...utils import TorchioTestCase


class TestRandomBiasField(TorchioTestCase):
    """Tests for `RandomBiasField`."""
    def test_no_bias(self):
        transform = RandomBiasField(coefficients=0.)
        transformed = transform(self.sample)
        self.assertTensorAlmostEqual(self.sample.t1.data, transformed.t1.data)

    def test_with_bias(self):
        transform = RandomBiasField(coefficients=0.1)
        transformed = transform(self.sample)
        self.assertTensorNotEqual(self.sample.t1.data, transformed.t1.data)

    def test_wrong_coefficient_type(self):
        with self.assertRaises(ValueError):
            RandomBiasField(coefficients='wrong')

    def test_negative_order(self):
        with self.assertRaises(ValueError):
            RandomBiasField(order=-1)

    def test_wrong_order_type(self):
        with self.assertRaises(TypeError):
            RandomBiasField(order='wrong')
