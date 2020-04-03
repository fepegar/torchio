from numpy.testing import assert_array_equal
import torchio
from torchio import DATA, AFFINE
from ...utils import TorchioTestCase


class TestCropOrPad(TorchioTestCase):
    """Tests for `CropOrPad`."""
    def test_no_changes(self):
        shape = self.sample['t1'][DATA].shape[1:]
        transform = torchio.transforms.CropOrPad(shape)
        transformed = transform(self.sample)
        assert_array_equal(self.sample['t1'][DATA], transformed['t1'][DATA])
        assert_array_equal(self.sample['t1'][AFFINE], transformed['t1'][AFFINE])

    def test_different_shape(self):
        shape = self.sample['t1'][DATA].shape[1:]
        target_shape = 9, 21, 30
        transform = torchio.transforms.CropOrPad(target_shape)
        transformed = transform(self.sample)
        result_shape = transformed['t1'][DATA].shape[1:]
        self.assertNotEqual(shape, result_shape)

    def test_shape_right(self):
        target_shape = 9, 21, 30
        transform = torchio.transforms.CropOrPad(target_shape)
        transformed = transform(self.sample)
        result_shape = transformed['t1'][DATA].shape[1:]
        self.assertEqual(target_shape, result_shape)

    def test_shape_negative(self):
        with self.assertRaises(ValueError):
            torchio.transforms.CropOrPad(-1)

    def test_shape_float(self):
        with self.assertRaises(ValueError):
            torchio.transforms.CropOrPad(2.5)

    def test_shape_string(self):
        with self.assertRaises(ValueError):
            torchio.transforms.CropOrPad('')

    def test_shape_one(self):
        transform = torchio.transforms.CropOrPad(1)
        transformed = transform(self.sample)
        result_shape = transformed['t1'][DATA].shape[1:]
        self.assertEqual((1, 1, 1), result_shape)

    def test_wrong_mask_name(self):
        cop = torchio.transforms.CropOrPad(1, mask_name='wrong')
        with self.assertRaises(KeyError):
            cop(self.sample)

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            torchio.transforms.CenterCropOrPad(1)

    def test_empty_mask(self):
        pass

    def test_center_mask(self):
        """The mask bounding box and the input image have the same center"""
        target_shape = 8, 22, 30
        transform_center = torchio.transforms.CropOrPad(target_shape)
        transform_mask = torchio.transforms.CropOrPad(
            target_shape, mask_name='label')
        mask = self.sample['label'][DATA]
        mask *= 0
        mask[0, 4:6, 9:11, 14:16] = 1
        transformed_center = transform_center(self.sample)
        transformed_mask = transform_mask(self.sample)
        zipped = zip(transformed_center.values(), transformed_mask.values())
        for image_center, image_mask in zipped:
            assert_array_equal(
                image_center[DATA], image_mask[DATA],
                'Data is different after cropping',
            )
            assert_array_equal(
                image_center[AFFINE], image_mask[AFFINE],
                'Physical position is different after cropping',
            )
