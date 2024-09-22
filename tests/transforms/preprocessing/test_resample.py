import numpy as np
import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestResample(TorchioTestCase):
    """Tests for `Resample`."""

    def test_spacing(self):
        # Should this raise an error if sizes are different?
        spacing = 2
        transform = tio.Resample(spacing)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            assert image.spacing == 3 * (spacing,)

    def test_reference_name(self):
        subject = self.get_inconsistent_shape_subject()
        reference_name = 't1'
        transform = tio.Resample(reference_name)
        transformed = transform(subject)
        reference_image = subject[reference_name]
        for image in transformed.get_images(intensity_only=False):
            assert reference_image.shape == image.shape
            self.assert_tensor_almost_equal(
                reference_image.affine,
                image.affine,
            )

    def test_affine(self):
        spacing = 1
        affine_name = 'pre_affine'
        transform = tio.Resample(spacing, pre_affine_name=affine_name)
        transformed = transform(self.sample_subject)
        for image in transformed.values():
            if affine_name in image:
                target_affine = np.eye(4)
                target_affine[:3, 3] = 10, 0, -0.1
                self.assert_tensor_almost_equal(image.affine, target_affine)
            else:
                self.assert_tensor_equal(image.affine, np.eye(4))

    def test_missing_affine(self):
        transform = tio.Resample(1, pre_affine_name='missing')
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_reference_path(self):
        reference_image, reference_path = self.get_reference_image_and_path()
        transform = tio.Resample(reference_path)
        transformed = transform(self.sample_subject)
        for image in transformed.values():
            assert reference_image.shape == image.shape
            self.assert_tensor_almost_equal(
                reference_image.affine,
                image.affine,
            )

    def test_wrong_spacing_length(self):
        with pytest.raises(RuntimeError):
            tio.Resample((1, 2))(self.sample_subject)

    def test_wrong_spacing_value(self):
        with pytest.raises(ValueError):
            tio.Resample(0)(self.sample_subject)

    def test_wrong_target_type(self):
        with pytest.raises(RuntimeError):
            tio.Resample(None)(self.sample_subject)

    def test_missing_reference(self):
        transform = tio.Resample('missing')
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_2d(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 1))
        transform = tio.Resample(0.5)
        shape = transform(image).shape
        assert shape == (1, 4, 6, 1)

    def test_input_list(self):
        tio.Resample([1, 2, 3])(self.sample_subject)

    def test_input_array(self):
        resample = tio.Resample(np.asarray([1, 2, 3]))
        resample(self.sample_subject)

    def test_image_target(self):
        tio.Resample(self.sample_subject.t1)(self.sample_subject)

    def test_bad_affine(self):
        shape = 1, 2, 3
        affine = np.eye(3)
        target = shape, affine
        transform = tio.Resample(target)
        with pytest.raises(RuntimeError):
            transform(self.sample_subject)
