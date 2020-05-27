import numpy as np
from numpy.testing import assert_array_equal
from torchio import DATA, AFFINE
from torchio.transforms import Resample
from torchio.utils import nib_to_sitk
from ...utils import TorchioTestCase


class TestResample(TorchioTestCase):
    """Tests for `Resample`."""
    def test_spacing(self):
        # Should this raise an error if sizes are different?
        spacing = 2
        transform = Resample(spacing)
        transformed = transform(self.sample)
        for image_dict in transformed.values():
            image = nib_to_sitk(image_dict[DATA], image_dict[AFFINE])
            self.assertEqual(image.GetSpacing(), 3 * (spacing,))

    def test_reference_name(self):
        sample = self.get_inconsistent_sample()
        reference_name = 't1'
        transform = Resample(reference_name)
        transformed = transform(sample)
        ref_image_dict = sample[reference_name]
        for image_dict in transformed.values():
            self.assertEqual(
                ref_image_dict.shape, image_dict.shape)
            assert_array_equal(ref_image_dict[AFFINE], image_dict[AFFINE])

    def test_affine(self):
        spacing = 1
        affine_name = 'pre_affine'
        transform = Resample(spacing, pre_affine_name=affine_name)
        transformed = transform(self.sample)
        for image_dict in transformed.values():
            if affine_name in image_dict.keys():
                new_affine = np.eye(4)
                new_affine[0, 3] = 10
                assert_array_equal(image_dict[AFFINE], new_affine)
            else:
                assert_array_equal(image_dict[AFFINE], np.eye(4))

    def test_missing_affine(self):
        transform = Resample(1, pre_affine_name='missing')
        with self.assertRaises(ValueError):
            transform(self.sample)

    def test_reference_path(self):
        reference_image, reference_path = self.get_reference_image_and_path()
        transform = Resample(reference_path)
        transformed = transform(self.sample)
        ref_data, ref_affine = reference_image.load()
        for image_dict in transformed.values():
            self.assertEqual(
                ref_data.shape, image_dict.shape)
            assert_array_equal(ref_affine, image_dict[AFFINE])

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
            transform(self.sample)
