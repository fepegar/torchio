import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestEnsureShapeMultiple(TorchioTestCase):
    def test_bad_method(self):
        with pytest.raises(ValueError):
            tio.EnsureShapeMultiple(1, method='bad')

    def test_pad(self):
        sample_t1 = self.sample_subject.t1
        assert sample_t1.shape == (1, 10, 20, 30)
        transform = tio.EnsureShapeMultiple(4, method='pad')
        transformed = transform(sample_t1)
        assert transformed.shape == (1, 12, 20, 32)

    def test_crop(self):
        sample_t1 = self.sample_subject.t1
        assert sample_t1.shape == (1, 10, 20, 30)
        transform = tio.EnsureShapeMultiple(4, method='crop')
        transformed = transform(sample_t1)
        assert transformed.shape == (1, 8, 20, 28)

    def test_2d(self):
        sample_t1 = self.sample_subject.t1
        sample_2d = sample_t1.data[..., :1]
        assert sample_2d.shape == (1, 10, 20, 1)
        transform = tio.EnsureShapeMultiple(4, method='crop')
        transformed = transform(sample_2d)
        assert transformed.shape == (1, 8, 20, 1)
