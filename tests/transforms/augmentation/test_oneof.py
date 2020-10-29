from torchio.transforms import OneOf, RandomAffine, RandomElasticDeformation
from ...utils import TorchioTestCase


class TestOneOf(TorchioTestCase):
    """Tests for `OneOf`."""
    def test_wrong_input_type(self):
        with self.assertRaises(ValueError):
            OneOf(1)

    def test_negative_probabilities(self):
        with self.assertRaises(ValueError):
            OneOf({RandomAffine(): -1, RandomElasticDeformation(): 1})

    def test_zero_probabilities(self):
        with self.assertRaises(ValueError):
            OneOf({RandomAffine(): 0, RandomElasticDeformation(): 0})

    def test_not_transform(self):
        with self.assertRaises(ValueError):
            OneOf({RandomAffine: 1, RandomElasticDeformation: 2})

    def test_one_of(self):
        transform = OneOf({
            RandomAffine(): 0.2,
            RandomElasticDeformation(max_displacement=0.5): 0.8,
        })
        transform(self.sample_subject)
