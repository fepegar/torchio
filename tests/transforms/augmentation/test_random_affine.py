from torchio.transforms import RandomAffine
from ...utils import TorchioTestCase


class TestRandomAffine(TorchioTestCase):
    """Tests for `RandomAffine`."""
    def setUp(self):
        # Set image origin far from center
        super().setUp()
        affine = self.sample.t1.affine
        affine[:3, 3] = 1e5

    def test_rotation_image(self):
        # Rotation around image center
        transform = RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample)
        total = transformed.t1.data.sum()
        self.assertNotEqual(total, 0)

    def test_rotation_origin(self):
        # Rotation around far away point, image should be empty
        transform = RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='origin',
        )
        transformed = transform(self.sample)
        total = transformed.t1.data.sum()
        self.assertEqual(total, 0)
