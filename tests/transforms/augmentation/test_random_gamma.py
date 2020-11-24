import torch
from torchio import RandomGamma
from ...utils import TorchioTestCase


class TestRandomGamma(TorchioTestCase):
    """Tests for `RandomGamma`."""
    def test_with_zero_gamma(self):
        transform = RandomGamma(log_gamma=0)
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(self.sample_subject.t1.data, transformed.t1.data)

    def test_with_non_zero_gamma(self):
        transform = RandomGamma(log_gamma=(0.1, 0.3))
        transformed = transform(self.sample_subject)
        self.assertTensorNotEqual(self.sample_subject.t1.data, transformed.t1.data)

    def test_with_high_gamma(self):
        transform = RandomGamma(log_gamma=(100, 100))
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(
            self.sample_subject.t1.data == 1, transformed.t1.data
        )

    def test_with_low_gamma(self):
        transform = RandomGamma(log_gamma=(-100, -100))
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(
            self.sample_subject.t1.data > 0, transformed.t1.data
        )

    def test_wrong_gamma_type(self):
        with self.assertRaises(ValueError):
            RandomGamma(log_gamma='wrong')

    def test_negative_values(self):
        with self.assertWarns(RuntimeWarning):
            RandomGamma()(torch.rand(1, 3, 3, 3) - 1)
