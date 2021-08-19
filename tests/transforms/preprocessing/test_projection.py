import torchio as tio
from ...utils import TorchioTestCase


class TestProjection(TorchioTestCase):
    """Tests for :class:`tio.Projection` class."""

    def test_default_projection(self):
        transform = tio.Projection('S')
        transformed = transform(self.sample_subject)
        axis_index = self.sample_subject.t1.axis_name_to_index('S')
        self.assertEqual(transformed.t1.shape[axis_index], 1)

    def test_invalid_axis_name(self):
        with self.assertRaises(ValueError):
            transform = tio.Projection('123')
            transform(self.sample_subject)

    def test_projection_type_invalid_input(self):
        with self.assertRaises(ValueError):
            tio.Projection('S', projection_type='bad')

    def test_q_is_required(self):
        with self.assertRaises(ValueError):
            tio.Projection('S', projection_type='quantile')

    def test_q_is_invalid(self):
        with self.assertRaises(ValueError):
            tio.Projection('S', projection_type='quantile', q=2)

    def test_full_slabs_only(self):
        sub = tio.datasets.Colin27()
        transform1 = tio.Projection('S', slab_thickness=100, stride=100)
        transform2 = tio.Projection(
            'S', slab_thickness=100, stride=100, full_slabs_only=False)
        transformed1 = transform1(sub)
        transformed2 = transform2(sub)
        axis_index = sub.t1.axis_name_to_index('S')
        self.assertEqual(transformed1.t1.shape[axis_index], 1)
        self.assertEqual(transformed2.t1.shape[axis_index], 2)

    def test_maximum_intensity_projection(self):
        transform = tio.Projection('S', projection_type='max')
        transform(self.sample_subject)

    def test_minimum_intensity_projection(self):
        transform = tio.Projection('S', projection_type='min')
        transform(self.sample_subject)

    def test_mean_intensity_projection(self):
        transform = tio.Projection('S', projection_type='mean')
        transform(self.sample_subject)

    def test_median_intensity_projection(self):
        transform = tio.Projection('S', projection_type='median')
        transform(self.sample_subject)

    def test_quantile_intensity_projection(self):
        transform = tio.Projection('S', projection_type='quantile', q=0.75)
        transform(self.sample_subject)

    def test_very_large_slab_thickness(self):
        transform = tio.Projection('S', slab_thickness=1e6)
        transform(self.sample_subject)
