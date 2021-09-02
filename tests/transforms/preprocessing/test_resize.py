import torch
import numpy as np
import torchio as tio
from ...utils import TorchioTestCase


class TestResize(TorchioTestCase):
    """Tests for `Resize`."""
    def test_one_dim(self):
        target_shape = 5
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            self.assertEqual(image.spatial_shape, 3 * (target_shape,))

    def test_all_dims(self):
        target_shape = 11, 6, 7
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            self.assertEqual(image.spatial_shape, target_shape)

    def test_fix_shape(self):
        # We use values that are known to need cropping
        tensor = torch.rand(1, 8, 180, 320)
        affine = np.diag((5, 1, 1, 1))
        im = tio.ScalarImage(tensor=tensor, affine=affine)
        target = 12
        with self.assertWarns(UserWarning):
            result = tio.Resize(target)(im)
        self.assertEqual(result.spatial_shape, 3 * (target,))
