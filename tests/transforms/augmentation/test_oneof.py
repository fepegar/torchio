import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestOneOf(TorchioTestCase):
    """Tests for `OneOf`."""

    def test_wrong_input_type(self):
        with pytest.raises(ValueError):
            tio.OneOf(1)

    def test_negative_probabilities(self):
        transforms = {
            tio.RandomAffine(): -1,
            tio.RandomElasticDeformation(): 1,
        }
        with pytest.raises(ValueError):
            tio.OneOf(transforms)

    def test_zero_probabilities(self):
        with pytest.raises(ValueError):
            transforms = {
                tio.RandomAffine(): 0,
                tio.RandomElasticDeformation(): 0,
            }
            tio.OneOf(transforms)

    def test_not_transform(self):
        with pytest.raises(ValueError):
            tio.OneOf({tio.RandomAffine: 1, tio.RandomElasticDeformation: 2})

    def test_one_of(self):
        transform = tio.OneOf(
            {
                tio.RandomAffine(): 0.2,
                tio.RandomElasticDeformation(max_displacement=0.5): 0.8,
            }
        )
        transform(self.sample_subject)
