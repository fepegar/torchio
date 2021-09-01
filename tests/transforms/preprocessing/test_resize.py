import torchio as tio
from ...utils import TorchioTestCase


class TestResize(TorchioTestCase):
    """Tests for `Resize`."""
    def test_one_dim(self):
        target_shape = 5
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            self.assertEqual(image.spatial_shape, 3 * (target_shape,))

    def test_all_dims(self):
        target_shape = 11, 6, 7
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            self.assertEqual(image.spatial_shape, target_shape)
