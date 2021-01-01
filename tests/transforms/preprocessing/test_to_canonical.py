import numpy as np
from torchio.transforms import ToCanonical
from ...utils import TorchioTestCase


class TestToCanonical(TorchioTestCase):
    """Tests for :class:`ToCanonical` class."""

    def test_no_changes(self):
        transform = ToCanonical()
        transformed = transform(self.sample_subject)
        self.assertTensorEqual(
            transformed.t1.data,
            self.sample_subject.t1.data,
        )
        self.assertTensorEqual(
            transformed.t1.affine,
            self.sample_subject.t1.affine,
        )

    def test_las_to_ras(self):
        self.sample_subject.t1.affine[0, 0] = -1  # Change orientation to 'LAS'
        transform = ToCanonical()
        transformed = transform(self.sample_subject)
        self.assertEqual(transformed.t1.orientation, ('R', 'A', 'S'))
        self.assertTensorAlmostEqual(
            transformed.t1.data,
            self.sample_subject.t1.data.numpy()[:, ::-1, :, :]
        )

        fixture = np.eye(4)
        fixture[0, -1] = -self.sample_subject.t1.spatial_shape[0] + 1
        self.assertTensorEqual(transformed.t1.affine, fixture)
