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

    def test_reference(self):
        sample = self.get_inconsistent_sample()
        reference_name = 't1'
        transform = Resample(reference_name)
        transformed = transform(sample)
        ref_image_dict = sample[reference_name]
        for image_dict in transformed.values():
            self.assertEqual(
                ref_image_dict[DATA].shape, image_dict[DATA].shape)
            assert_array_equal(ref_image_dict[AFFINE], image_dict[AFFINE])

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
