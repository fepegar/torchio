import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestZNormalization(TorchioTestCase):
    """Tests for :class:`ZNormalization` class."""

    def test_z_normalization(self):
        transform = tio.ZNormalization()
        transformed = transform(self.sample_subject)
        assert float(transformed.t1.data.mean()) == pytest.approx(0, abs=1e-6)
        assert float(transformed.t1.data.std()) == pytest.approx(1)

    def test_no_std(self):
        image = tio.ScalarImage(tensor=torch.ones(1, 2, 2, 2))
        with pytest.raises(RuntimeError):
            tio.ZNormalization()(image)

    def test_dtype(self):
        # https://github.com/TorchIO-project/torchio/issues/407
        tensor_int = (100 * torch.rand(1, 2, 3, 4)).byte()
        transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        transform(tensor_int)
        transform = tio.ZNormalization()
        transform(tensor_int)
