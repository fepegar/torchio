import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomBiasField(TorchioTestCase):
    def test_no_bias(self):
        transform = tio.RandomBiasField(coefficients=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_bias(self):
        transform = tio.RandomBiasField(coefficients=0.1)
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_wrong_coefficient_type(self):
        with pytest.raises(ValueError):
            tio.RandomBiasField(coefficients='wrong')

    def test_negative_order(self):
        with pytest.raises(ValueError):
            tio.RandomBiasField(order=-1)

    def test_wrong_order_type(self):
        with pytest.raises(TypeError):
            tio.RandomBiasField(order='wrong')

    def test_small_image(self):
        # https://github.com/TorchIO-project/torchio/issues/300
        tio.RandomBiasField()(torch.rand(1, 2, 3, 4))
