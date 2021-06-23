from torchio.transforms import CopyAffine
from ...utils import TorchioTestCase
import torchio as tio
import numpy as np
import torch


class TestCopyAffine(TorchioTestCase):
    """Tests for `CopyAffine`."""

    def test_missing_reference(self):
        transform = CopyAffine(target_key='missing')
        with self.assertRaises(KeyError):
            transform(self.sample_subject)

    def test_wrong_target_type(self):
        transform = CopyAffine(target_key=[1])
        with self.assertRaises(TypeError):
            transform(self.sample_subject)

    def test_same_affine(self):
        image = tio.ScalarImage(tensor=torch.rand(2, 2, 2, 2))
        mask = tio.LabelMap(tensor=torch.rand(2, 2, 2, 2))
        mask.affine *= 1.1
        subject = tio.Subject(t1=image, mask=mask)
        transform = CopyAffine('t1')
        transformed = transform(subject)
        self.assertTrue(np.array_equal(transformed['t1'].affine,
                                       transformed['mask'].affine))
