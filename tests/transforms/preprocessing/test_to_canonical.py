import numpy as np
from torchio.transforms import ToCanonical
from ...utils import TorchioTestCase


class TestToCanonical(TorchioTestCase):
    """Tests for :py:class:`ToCanonical` class."""

    def test_no_changes(self):
        transform = ToCanonical()
        transformed = transform(self.sample)
        self.assertTensorEqual(transformed.t1.data, self.sample.t1.data)
        self.assertTensorEqual(transformed.t1.affine, self.sample.t1.affine)

    def test_LAS_to_RAS(self):
        self.sample.t1.affine[0, 0] = -1    # Change orientation to 'LAS'
        transform = ToCanonical()
        transformed = transform(self.sample)
        self.assertEqual(transformed.t1.orientation, ('R', 'A', 'S'))
        self.assertTensorEqual(
            transformed.t1.data,
            self.sample.t1.data.numpy()[:, ::-1, :, :]
        )

        fixture = np.eye(4)
        fixture[0, -1] = -self.sample.t1.spatial_shape[0] + 1
        self.assertTensorEqual(transformed.t1.affine, fixture)
