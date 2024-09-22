import numpy as np
import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestCopyAffine(TorchioTestCase):
    """Tests for `CopyAffine`."""

    def test_missing_reference(self):
        transform = tio.CopyAffine(target='missing')
        with pytest.raises(RuntimeError):
            transform(self.sample_subject)

    def test_wrong_target_type(self):
        with pytest.raises(ValueError):
            tio.CopyAffine(target=[1])

    def test_same_affine(self):
        image = tio.ScalarImage(tensor=torch.rand(2, 2, 2, 2))
        mask = tio.LabelMap(tensor=torch.rand(2, 2, 2, 2))
        mask.affine *= 1.1
        subject = tio.Subject(t1=image, mask=mask)
        transform = tio.CopyAffine('t1')
        transformed = transform(subject)
        self.assert_tensor_equal(
            transformed['t1'].affine,
            transformed['mask'].affine,
        )

    def test_before_loading(self):
        affine = 2 * np.eye(4)
        image = tio.ScalarImage(tensor=torch.rand(2, 2, 2, 2), affine=affine)
        mask = tio.LabelMap(self.get_image_path('mask'))
        subject = tio.Subject(t1=image, mask=mask)
        transform = tio.CopyAffine('t1')
        transformed = transform(subject)
        self.assert_tensor_equal(
            transformed['t1'].affine,
            transformed['mask'].affine,
        )
