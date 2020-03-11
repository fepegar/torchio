from copy import deepcopy
import torch
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
