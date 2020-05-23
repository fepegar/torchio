import torch
from torchio import DATA, LABEL
from torchio.transforms import Lambda
from ..utils import TorchioTestCase


class TestLambda(TorchioTestCase):
    """Tests for :py:class:`Lambda` class."""

    def test_wrong_return_type(self):
        transform = Lambda(lambda x: 'Not a tensor')
        with self.assertRaises(ValueError):
            transform(self.sample)

    def test_wrong_return_data_type(self):
        transform = Lambda(lambda x: torch.rand(1) > 0)
        with self.assertRaises(ValueError):
            transform(self.sample)

    def test_wrong_return_shape(self):
        transform = Lambda(lambda x: torch.rand(1))
        with self.assertRaises(ValueError):
            transform(self.sample)

    def test_lambda(self):
        transform = Lambda(lambda x: x + 1)
        transformed = transform(self.sample)
        assert torch.all(torch.eq(
            transformed['t1'][DATA], self.sample['t1'][DATA] + 1))
        assert torch.all(torch.eq(
            transformed['t2'][DATA], self.sample['t2'][DATA] + 1))
        assert torch.all(torch.eq(
            transformed['label'][DATA], self.sample['label'][DATA] + 1))

    def test_image_types(self):
        transform = Lambda(lambda x: x + 1, types_to_apply=[LABEL])
        transformed = transform(self.sample)
        assert torch.all(torch.eq(
            transformed['t1'][DATA], self.sample['t1'][DATA]))
        assert torch.all(torch.eq(
            transformed['t2'][DATA], self.sample['t2'][DATA]))
        assert torch.all(torch.eq(
            transformed['label'][DATA], self.sample['label'][DATA] + 1))
