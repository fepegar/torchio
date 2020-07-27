from torchio import RandomMotion
from ...utils import TorchioTestCase
from numpy.testing import assert_array_equal


class TestRandomMotion(TorchioTestCase):
    """Tests for `RandomMotion`."""
    def test_bad_num_transforms_value(self):
        with self.assertRaises(ValueError):
            RandomMotion(num_transforms=0)

    def test_no_movement(self):
        transform = RandomMotion(
            degrees=0,
            translation=0,
            num_transforms=1
        )
        transformed = transform(self.sample)
        assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_with_movement(self):
        transform = RandomMotion(
            num_transforms=1
        )
        transformed = transform(self.sample)
        with self.assertRaises(AssertionError):
            assert_array_equal(self.sample.t1.data, transformed.t1.data)

    def test_negative_degrees(self):
        with self.assertRaises(ValueError):
            RandomMotion(degrees=-10)

    def test_wrong_degrees_type(self):
        with self.assertRaises(ValueError):
            RandomMotion(degrees='wrong')

    def test_negative_translation(self):
        with self.assertRaises(ValueError):
            RandomMotion(translation=-10)

    def test_wrong_translation_type(self):
        with self.assertRaises(ValueError):
            RandomMotion(translation='wrong')

    def test_wrong_image_interpolation_type(self):
        with self.assertRaises(TypeError):
            RandomMotion(image_interpolation=0)

    def test_wrong_image_interpolation_value(self):
        with self.assertRaises(AttributeError):
            RandomMotion(image_interpolation='wrong')
