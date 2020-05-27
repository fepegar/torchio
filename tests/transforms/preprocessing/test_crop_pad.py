import warnings
import numpy as np
from numpy.testing import assert_array_equal
from torchio.transforms import CropOrPad, CenterCropOrPad
from torchio import DATA, AFFINE
from ...utils import TorchioTestCase


class TestCropOrPad(TorchioTestCase):
    """Tests for `CropOrPad`."""
    def test_no_changes(self):
        sample_t1 = self.sample['t1']
        shape = sample_t1.spatial_shape
        transform = CropOrPad(shape)
        transformed = transform(self.sample)
        assert_array_equal(sample_t1[DATA], transformed['t1'][DATA])
        assert_array_equal(sample_t1[AFFINE], transformed['t1'][AFFINE])

    def test_no_changes_mask(self):
        sample_t1 = self.sample['t1']
        sample_mask = self.sample['label'][DATA]
        sample_mask *= 0
        shape = sample_t1.spatial_shape
        transform = CropOrPad(shape, mask_name='label')
        with self.assertWarns(UserWarning):
            transformed = transform(self.sample)
        for key in transformed:
            image_dict = self.sample[key]
            assert_array_equal(image_dict[DATA], transformed[key][DATA])
            assert_array_equal(image_dict[AFFINE], transformed[key][AFFINE])

    def test_different_shape(self):
        shape = self.sample['t1'].spatial_shape
        target_shape = 9, 21, 30
        transform = CropOrPad(target_shape)
        transformed = transform(self.sample)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertNotEqual(shape, result_shape)

    def test_shape_right(self):
        target_shape = 9, 21, 30
        transform = CropOrPad(target_shape)
        transformed = transform(self.sample)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape)

    def test_only_pad(self):
        target_shape = 11, 22, 30
        transform = CropOrPad(target_shape)
        transformed = transform(self.sample)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape)

    def test_only_crop(self):
        target_shape = 9, 18, 30
        transform = CropOrPad(target_shape)
        transformed = transform(self.sample)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape)

    def test_shape_negative(self):
        with self.assertRaises(ValueError):
            CropOrPad(-1)

    def test_shape_float(self):
        with self.assertRaises(ValueError):
            CropOrPad(2.5)

    def test_shape_string(self):
        with self.assertRaises(ValueError):
            CropOrPad('')

    def test_shape_one(self):
        transform = CropOrPad(1)
        transformed = transform(self.sample)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual((1, 1, 1), result_shape)

    def test_wrong_mask_name(self):
        cop = CropOrPad(1, mask_name='wrong')
        with self.assertWarns(UserWarning):
            cop(self.sample)

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            CenterCropOrPad(1)

    def test_empty_mask(self):
        target_shape = 8, 22, 30
        transform = CropOrPad(target_shape, mask_name='label')
        mask = self.sample['label'][DATA]
        mask *= 0
        with self.assertWarns(UserWarning):
            transform(self.sample)

    def test_mask_only_pad(self):
        target_shape = 11, 22, 30
        transform = CropOrPad(target_shape, mask_name='label')
        mask = self.sample['label'][DATA]
        mask *= 0
        mask [0, 4:6, 5:8, 3:7] = 1
        transformed = transform(self.sample)
        shapes = []
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            shapes.append(result_shape)
        set_shapes = set(shapes)
        message = f'Images have different shapes: {set_shapes}'
        assert len(set_shapes) == 1, message
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape,
                f'Wrong shape for image: {key}',
            )

    def test_mask_only_crop(self):
        target_shape = 9, 18, 30
        transform = CropOrPad(target_shape, mask_name='label')
        mask = self.sample['label'][DATA]
        mask *= 0
        mask [0, 4:6, 5:8, 3:7] = 1
        transformed = transform(self.sample)
        shapes = []
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            shapes.append(result_shape)
        set_shapes = set(shapes)
        message = f'Images have different shapes: {set_shapes}'
        assert len(set_shapes) == 1, message
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertEqual(target_shape, result_shape,
                f'Wrong shape for image: {key}',
            )

    def test_center_mask(self):
        """The mask bounding box and the input image have the same center"""
        target_shape = 8, 22, 30
        transform_center = CropOrPad(target_shape)
        transform_mask = CropOrPad(target_shape, mask_name='label')
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

    def test_mask_corners(self):
        """The mask bounding box and the input image have the same center"""
        target_shape = 8, 22, 30
        transform_center = CropOrPad(target_shape)
        transform_mask = CropOrPad(
            target_shape, mask_name='label')
        mask = self.sample['label'][DATA]
        mask *= 0
        mask[0, 0, 0, 0] = 1
        mask[0, -1, -1, -1] = 1
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

    def test_mask_origin(self):
        target_shape = 7, 21, 29
        center_voxel = np.floor(np.array(target_shape) / 2).astype(int)
        transform_center = CropOrPad(target_shape)
        transform_mask = CropOrPad(
            target_shape, mask_name='label')
        mask = self.sample['label'][DATA]
        mask *= 0
        mask[0, 0, 0, 0] = 1
        transformed_center = transform_center(self.sample)
        transformed_mask = transform_mask(self.sample)
        zipped = zip(transformed_center.values(), transformed_mask.values())
        for image_center, image_mask in zipped:
            # Arrays are different
            assert not np.array_equal(image_center[DATA], image_mask[DATA])
            # Rotation matrix doesn't change
            center_rotation = image_center[AFFINE][:3, :3]
            mask_rotation = image_mask[AFFINE][:3, :3]
            assert_array_equal(center_rotation, mask_rotation)
            # Origin does change
            center_origin = image_center[AFFINE][:3, 3]
            mask_origin = image_mask[AFFINE][:3, 3]
            assert not np.array_equal(center_origin, mask_origin)
            # Voxel at origin is center of transformed image
            origin_value = image_center[DATA][0, 0, 0, 0]
            i, j, k = center_voxel
            transformed_value = image_mask[DATA][0, i, j, k]
            self.assertEqual(origin_value, transformed_value)
