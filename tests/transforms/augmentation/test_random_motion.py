import pytest

from torchio import RandomMotion

from ...utils import TorchioTestCase


class TestRandomMotion(TorchioTestCase):
    """Tests for `RandomMotion`."""

    def test_bad_num_transforms_value(self):
        with pytest.raises(ValueError):
            RandomMotion(num_transforms=0)

    def test_no_movement(self):
        transform = RandomMotion(
            degrees=0,
            translation=0,
            num_transforms=1,
        )
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
            atol=1e-4,
            rtol=0,
        )

    def test_with_movement(self):
        transform = RandomMotion(
            num_transforms=1,
        )
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_negative_degrees(self):
        with pytest.raises(ValueError):
            RandomMotion(degrees=-10)

    def test_wrong_degrees_type(self):
        with pytest.raises(ValueError):
            RandomMotion(degrees='wrong')

    def test_negative_translation(self):
        with pytest.raises(ValueError):
            RandomMotion(translation=-10)

    def test_wrong_translation_type(self):
        with pytest.raises(ValueError):
            RandomMotion(translation='wrong')

    def test_wrong_image_interpolation_type(self):
        with pytest.raises(TypeError):
            RandomMotion(image_interpolation=0)

    def test_wrong_image_interpolation_value(self):
        with pytest.raises(ValueError):
            RandomMotion(image_interpolation='wrong')
