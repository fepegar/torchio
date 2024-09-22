import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomSwap(TorchioTestCase):
    def test_no_swap(self):
        transform = tio.RandomSwap(patch_size=5, num_iterations=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_swap(self):
        transform = tio.RandomSwap(patch_size=5)
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_wrong_num_iterations_type(self):
        with pytest.raises(TypeError):
            tio.RandomSwap(num_iterations='wrong')

    def test_negative_num_iterations(self):
        with pytest.raises(ValueError):
            tio.RandomSwap(num_iterations=-1)
