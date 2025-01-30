import numpy as np
import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestCropOrPad(TorchioTestCase):
    """Tests for `CropOrPad`."""

    def test_no_changes(self):
        sample_t1 = self.sample_subject['t1']
        shape = sample_t1.spatial_shape
        transform = tio.CropOrPad(shape)
        transformed = transform(self.sample_subject)
        self.assert_tensor_equal(sample_t1.data, transformed['t1'].data)
        self.assert_tensor_equal(sample_t1.affine, transformed['t1'].affine)

    def test_no_changes_mask(self):
        sample_t1 = self.sample_subject['t1']
        sample_mask = self.sample_subject['label'].data
        sample_mask *= 0
        shape = sample_t1.spatial_shape
        transform = tio.CropOrPad(shape, mask_name='label')
        with pytest.warns(RuntimeWarning):
            transformed = transform(self.sample_subject)
        for key in transformed:
            image = self.sample_subject[key]
            self.assert_tensor_equal(image.data, transformed[key].data)
            self.assert_tensor_equal(image.affine, transformed[key].affine)

    def test_different_shape(self):
        shape = self.sample_subject['t1'].spatial_shape
        target_shape = 9, 21, 30
        transform = tio.CropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            self.assertNotEqual(shape, result_shape)

    def test_shape_right(self):
        target_shape = 9, 21, 30
        transform = tio.CropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            assert target_shape == result_shape

    def test_only_pad(self):
        target_shape = 11, 22, 30
        transform = tio.CropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            assert target_shape == result_shape

    def test_only_crop(self):
        target_shape = 9, 18, 30
        transform = tio.CropOrPad(target_shape)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            assert target_shape == result_shape

    def test_shape_negative(self):
        with pytest.raises(ValueError):
            tio.CropOrPad(-1)

    def test_shape_float(self):
        with pytest.raises(ValueError):
            tio.CropOrPad(2.5)

    def test_shape_string(self):
        with pytest.raises(ValueError):
            tio.CropOrPad('')

    def test_shape_one(self):
        transform = tio.CropOrPad(1)
        transformed = transform(self.sample_subject)
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            assert result_shape == (1, 1, 1)

    def test_wrong_mask_name(self):
        cop = tio.CropOrPad(1, mask_name='wrong')
        with pytest.warns(RuntimeWarning):
            cop(self.sample_subject)

    def test_empty_mask(self):
        target_shape = 8, 22, 30
        transform = tio.CropOrPad(target_shape, mask_name='label')
        mask = self.sample_subject['label'].data
        mask *= 0
        with pytest.warns(RuntimeWarning):
            transform(self.sample_subject)

    def mask_only(self, target_shape):
        transform = tio.CropOrPad(target_shape, mask_name='label')
        mask = self.sample_subject['label'].data
        mask *= 0
        mask[0, 4:6, 5:8, 3:7] = 1
        transformed = transform(self.sample_subject)
        shapes = []
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            shapes.append(result_shape)
        set_shapes = set(shapes)
        message = f'Images have different shapes: {set_shapes}'
        assert len(set_shapes) == 1, message
        for key in transformed:
            result_shape = transformed[key].spatial_shape
            assert target_shape == result_shape, f'Wrong shape for image: {key}'

    def test_mask_only_pad(self):
        self.mask_only((11, 22, 30))

    def test_mask_only_crop(self):
        self.mask_only((9, 18, 30))

    def test_center_mask(self):
        """The mask bounding box and the input image have the same center."""
        target_shape = 8, 22, 30
        transform_center = tio.CropOrPad(target_shape)
        transform_mask = tio.CropOrPad(target_shape, mask_name='label')
        mask = self.sample_subject['label'].data
        mask *= 0
        mask[0, 4:6, 9:11, 14:16] = 1
        transformed_center = transform_center(self.sample_subject)
        transformed_mask = transform_mask(self.sample_subject)
        zipped = zip(transformed_center.values(), transformed_mask.values())
        for image_center, image_mask in zipped:
            self.assert_tensor_equal(
                image_center.data,
                image_mask.data,
                msg='Data is different after cropping',
            )
            self.assert_tensor_equal(
                image_center.affine,
                image_mask.affine,
                msg='Physical position is different after cropping',
            )

    def test_mask_corners(self):
        """The mask bounding box and the input image have the same center."""
        target_shape = 8, 22, 30
        transform_center = tio.CropOrPad(target_shape)
        transform_mask = tio.CropOrPad(
            target_shape,
            mask_name='label',
        )
        mask = self.sample_subject['label'].data
        mask *= 0
        mask[0, 0, 0, 0] = 1
        mask[0, -1, -1, -1] = 1
        transformed_center = transform_center(self.sample_subject)
        transformed_mask = transform_mask(self.sample_subject)
        zipped = zip(transformed_center.values(), transformed_mask.values())
        for image_center, image_mask in zipped:
            self.assert_tensor_equal(
                image_center.data,
                image_mask.data,
                msg='Data is different after cropping',
            )
            self.assert_tensor_equal(
                image_center.affine,
                image_mask.affine,
                msg='Physical position is different after cropping',
            )

    def test_2d(self):
        # https://github.com/TorchIO-project/torchio/issues/434
        image = np.random.rand(1, 16, 16, 1)
        mask = np.zeros_like(image, dtype=bool)
        mask[0, 7, 0] = True
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask),
        )
        transform = tio.CropOrPad((12, 12, 1), mask_name='mask')
        transformed = transform(subject)
        assert transformed.shape == (1, 12, 12, 1)

    def test_no_target_no_mask(self):
        with pytest.raises(ValueError):
            tio.CropOrPad()

    def test_labels_but_no_mask(self):
        with pytest.raises(ValueError):
            tio.CropOrPad(target_shape=(3, 4, 5), labels=[2, 3])

    def test_no_target(self):
        crop_with_mask = tio.CropOrPad(mask_name='label')
        crop_with_mask(self.sample_subject)

    def test_persistent_bounds_params(self):
        # https://github.com/TorchIO-project/torchio/issues/757
        shape = (1, 5, 5, 5)
        mask_a = np.zeros(shape)
        mask_a[0, 2, 2, 2] = 1
        mask_b = mask_a.copy()
        mask_b[0, 1:4, 1:4, 1:4] = 1
        tensor = np.ones(shape)
        image_a = tio.ScalarImage(tensor=tensor)
        mask_a = tio.LabelMap(tensor=mask_a)
        subject_a = tio.Subject(image=image_a, mask=mask_a)
        image_b = tio.ScalarImage(tensor=tensor)
        mask_b = tio.LabelMap(tensor=mask_b)
        subject_b = tio.Subject(image=image_b, mask=mask_b)
        crop = tio.CropOrPad(mask_name='mask')
        for _ in range(2):
            shape_a = crop(subject_a).image.shape
            shape_b = crop(subject_b).image.shape
            assert shape_a != shape_b
