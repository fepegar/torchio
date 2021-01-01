from torchio import RandomSwap
from ...utils import TorchioTestCase


class TestRandomSwap(TorchioTestCase):
    """Tests for `RandomSwap`."""
    def test_no_swap(self):
        transform = RandomSwap(patch_size=5, num_iterations=0)
        transformed = transform(self.sample_subject)
        self.assertTensorEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_swap(self):
        transform = RandomSwap(patch_size=5)
        transformed = transform(self.sample_subject)
        self.assertTensorNotEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_wrong_num_iterations_type(self):
        with self.assertRaises(TypeError):
            RandomSwap(num_iterations='wrong')

    def test_negative_num_iterations(self):
        with self.assertRaises(ValueError):
            RandomSwap(num_iterations=-1)
