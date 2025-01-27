import numpy as np
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestToCanonical(TorchioTestCase):
    def test_no_changes(self):
        transform = tio.ToCanonical()
        transformed = transform(self.sample_subject)
        self.assert_tensor_equal(
            transformed.t1.data,
            self.sample_subject.t1.data,
        )
        self.assert_tensor_equal(
            transformed.t1.affine,
            self.sample_subject.t1.affine,
        )

    def test_las_to_ras(self):
        self.sample_subject.t1.affine[0, 0] = -1  # Change orientation to 'LAS'
        transform = tio.ToCanonical()
        transformed = transform(self.sample_subject)
        assert transformed.t1.orientation == ('R', 'A', 'S')
        array_flip = self.sample_subject.t1.data.numpy()[:, ::-1, :, :].copy()
        self.assert_tensor_almost_equal(
            transformed.t1.data,
            torch.from_numpy(array_flip),
            check_stride=False,
        )

        fixture = np.eye(4)
        fixture[0, -1] = -self.sample_subject.t1.spatial_shape[0] + 1
        self.assert_tensor_equal(transformed.t1.affine, fixture)
