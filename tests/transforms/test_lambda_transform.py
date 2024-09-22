import pytest
import torch

from torchio import LABEL
from torchio.transforms import Lambda

from ..utils import TorchioTestCase


class TestLambda(TorchioTestCase):
    """Tests for :class:`Lambda` class."""

    def test_wrong_return_type(self):
        transform = Lambda(lambda x: 'Not a tensor')
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_wrong_return_data_type(self):
        transform = Lambda(lambda x: torch.rand(1) > 0)
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_wrong_return_shape(self):
        transform = Lambda(lambda x: torch.rand(1))
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_lambda(self):
        transform = Lambda(lambda x: x + 1)
        transformed = transform(self.sample_subject)
        assert torch.all(
            torch.eq(
                transformed.t1.data,
                self.sample_subject.t1.data + 1,
            ),
        )
        assert torch.all(
            torch.eq(
                transformed.t2.data,
                self.sample_subject.t2.data + 1,
            ),
        )
        assert torch.all(
            torch.eq(
                transformed.label.data,
                self.sample_subject.label.data + 1,
            ),
        )

    def test_image_types(self):
        transform = Lambda(lambda x: x + 1, types_to_apply=[LABEL])
        transformed = transform(self.sample_subject)
        assert torch.all(
            torch.eq(
                transformed.t1.data,
                self.sample_subject.t1.data,
            ),
        )
        assert torch.all(
            torch.eq(
                transformed.t2.data,
                self.sample_subject.t2.data,
            ),
        )
        assert torch.all(
            torch.eq(
                transformed.label.data,
                self.sample_subject.label.data + 1,
            ),
        )
