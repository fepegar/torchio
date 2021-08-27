import torchio as tio
from ...utils import TorchioTestCase


class TestProjection(TorchioTestCase):
    """Tests for :class:`tio.SlabProjection` class."""

    def test_default_projection(self):
        transform = tio.SlabProjection('S')
        transformed = transform(self.sample_subject)
        axis_index = self.sample_subject.t1.axis_name_to_index('S')
        self.assertEqual(transformed.t1.shape[axis_index], 1)

    def test_invalid_axis_name(self):
        with self.assertRaises(ValueError):
            transform = tio.SlabProjection('123')
            transform(self.sample_subject)

    def test_projection_type_invalid_input(self):
        with self.assertRaises(ValueError):
            tio.SlabProjection('S', projection_type='bad')

    def test_percentile_is_required(self):
        with self.assertRaises(TypeError):
            tio.SlabProjection('S', projection_type='percentile')

    def test_percentile_is_invalid(self):
        with self.assertRaises(ValueError):
            tio.SlabProjection(
                'S', projection_type='percentile', percentile=-1)

    def test_full_slabs_only(self):
        sub = tio.datasets.Colin27()
        transform1 = tio.SlabProjection('S', slab_thickness=100, stride=100)
        transform2 = tio.SlabProjection(
            'S', slab_thickness=100, stride=100, full_slabs_only=False)
        transformed1 = transform1(sub)
        transformed2 = transform2(sub)
        axis_index = sub.t1.axis_name_to_index('S')
        self.assertEqual(transformed1.t1.shape[axis_index], 1)
        self.assertEqual(transformed2.t1.shape[axis_index], 2)

    def test_maximum_intensity_projection(self):
        transform = tio.SlabProjection('S', projection_type='max')
        transform(self.sample_subject)

    def test_minimum_intensity_projection(self):
        transform = tio.SlabProjection('S', projection_type='min')
        transform(self.sample_subject)

    def test_mean_intensity_projection(self):
        transform = tio.SlabProjection('S', projection_type='mean')
        transform(self.sample_subject)

    def test_median_intensity_projection(self):
        transform = tio.SlabProjection('S', projection_type='median')
        transform(self.sample_subject)

    def test_percentile_intensity_projection(self):
        transform = tio.SlabProjection(
            'S', projection_type='percentile', percentile=75)
        transform(self.sample_subject)

    def test_very_large_slab_thickness(self):
        transform = tio.SlabProjection('S', slab_thickness=1e6)
        transformed = transform(self.sample_subject)
        axis_index = self.sample_subject.t1.axis_name_to_index('S')
        self.assertEqual(transformed.t1.shape[axis_index], 1)
