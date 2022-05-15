import torch
from torchio import RandomGamma
from ...utils import TorchioTestCase


class TestRandomGamma(TorchioTestCase):
    """Tests for `RandomGamma`."""
    def get_random_tensor_zero_one(self):
        return torch.rand(4, 5, 6, 7)

    def test_with_zero_gamma(self):
        transform = RandomGamma(log_gamma=0)
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assertTensorAlmostEqual(tensor, transformed)

    def test_with_non_zero_gamma(self):
        transform = RandomGamma(log_gamma=(0.1, 0.3))
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assertTensorNotEqual(tensor, transformed)

    def test_with_high_gamma(self):
        transform = RandomGamma(log_gamma=(100, 100))
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assertTensorAlmostEqual(
            tensor == 1, transformed
        )

    def test_with_low_gamma(self):
        transform = RandomGamma(log_gamma=(-100, -100))
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assertTensorAlmostEqual(
            tensor > 0, transformed
        )

    def test_wrong_gamma_type(self):
        with self.assertRaises(ValueError):
            RandomGamma(log_gamma='wrong')
