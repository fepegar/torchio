import torchio as tio
from ...utils import TorchioTestCase


class TestRandomCropOrPad(TorchioTestCase):
    """Tests for `RandomCropOrPad`."""
    def test_no_changes(self):
        sample_t1 = self.sample_subject['t1']
        shape = sample_t1.spatial_shape
        transform = tio.RandomCropOrPad(shape)
        transformed = transform(self.sample_subject)
        self.assertTensorEqual(sample_t1.data, transformed['t1'].data)
        self.assertTensorEqual(sample_t1.affine, transformed['t1'].affine)

    def test_different_shape(self):
        shape = self.sample_subject['t1'].spatial_shape
        target_shape = 9, 21, 30
        transform = tio.RandomCropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertNotEqual(shape, result_shape)

    def test_shape_right(self):
        target_shape = 9, 21, 30
        transform = tio.RandomCropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape)

    def test_only_pad(self):
        target_shape = 11, 22, 30
        transform = tio.RandomCropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape)

    def test_only_crop(self):
        target_shape = 9, 18, 30
        transform = tio.RandomCropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape)

    def test_shape_negative(self):
        with self.assertRaises(ValueError):
            tio.RandomCropOrPad(-1)

    def test_shape_float(self):
        with self.assertRaises(ValueError):
            tio.RandomCropOrPad(2.5)

    def test_shape_one(self):
        transform = tio.RandomCropOrPad(1)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual((1, 1, 1), result_shape)
