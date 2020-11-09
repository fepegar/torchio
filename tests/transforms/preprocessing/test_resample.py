import torch
import numpy as np
from torchio import ScalarImage
from torchio.transforms import Resample
from ...utils import TorchioTestCase


class TestResample(TorchioTestCase):
    """Tests for `Resample`."""
    def test_spacing(self):
        # Should this raise an error if sizes are different?
        spacing = 2
        transform = Resample(spacing)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            self.assertEqual(image.spacing, 3 * (spacing,))

    def test_reference_name(self):
        subject = self.get_inconsistent_shape_subject()
        reference_name = 't1'
        transform = Resample(reference_name)
        transformed = transform(subject)
        reference_image = subject[reference_name]
        for image in transformed.get_images(intensity_only=False):
            self.assertEqual(reference_image.shape, image.shape)
            self.assertTensorAlmostEqual(reference_image.affine, image.affine)

    def test_affine(self):
        spacing = 1
        affine_name = 'pre_affine'
        transform = Resample(spacing, pre_affine_name=affine_name)
        transformed = transform(self.sample_subject)
        for image in transformed.values():
            if affine_name in image:
                target_affine = np.eye(4)
                target_affine[:3, 3] = 10, 0, -0.1
                self.assertTensorAlmostEqual(image.affine, target_affine)
            else:
                self.assertTensorEqual(image.affine, np.eye(4))

    def test_missing_affine(self):
        transform = Resample(1, pre_affine_name='missing')
        with self.assertRaises(ValueError):
            transform(self.sample_subject)

    def test_reference_path(self):
        reference_image, reference_path = self.get_reference_image_and_path()
        transform = Resample(reference_path)
        transformed = transform(self.sample_subject)
        for image in transformed.values():
            self.assertEqual(reference_image.shape, image.shape)
            self.assertTensorAlmostEqual(reference_image.affine, image.affine)

    def test_wrong_spacing_length(self):
        with self.assertRaises(ValueError):
            Resample((1, 2))

    def test_wrong_spacing_value(self):
        with self.assertRaises(ValueError):
            Resample(0)

    def test_wrong_target_type(self):
        with self.assertRaises(ValueError):
            Resample(None)

    def test_missing_reference(self):
        transform = Resample('missing')
        with self.assertRaises(ValueError):
            transform(self.sample_subject)

    def test_2d(self):
        image = ScalarImage(tensor=torch.rand(1, 2, 3, 1))
        transform = Resample(0.5)
        shape = transform(image).shape
        self.assertEqual(shape, (1, 4, 6, 1))
