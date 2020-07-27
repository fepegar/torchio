import numpy as np
from torchio.transforms import RescaleIntensity
from ...utils import TorchioTestCase


class TestRescaleIntensity(TorchioTestCase):
    """Tests for :py:class:`RescaleIntensity` class."""

    def test_rescale_to_same_intentisy(self):
        min_t1 = float(self.sample.t1.data.min())
        max_t1 = float(self.sample.t1.data.max())
        transform = RescaleIntensity(out_min_max=(min_t1, max_t1))
        transformed = transform(self.sample)
        assert np.allclose(
            transformed.t1.data, self.sample.t1.data, rtol=0, atol=1e-06)

    def test_min_max(self):
        transform = RescaleIntensity(out_min_max=(0., 1.))
        transformed = transform(self.sample)
        self.assertEqual(transformed.t1.data.min(), 0.)
        self.assertEqual(transformed.t1.data.max(), 1.)

    def test_percentiles(self):
        low_quantile = np.percentile(self.sample.t1.data, 5)
        high_quantile = np.percentile(self.sample.t1.data, 95)
        low_indices = (self.sample.t1.data < low_quantile).nonzero(
            as_tuple=True)
        high_indices = (self.sample.t1.data > high_quantile).nonzero(
            as_tuple=True)
        transform = RescaleIntensity(out_min_max=(0., 1.), percentiles=(5, 95))
        transformed = transform(self.sample)
        assert (transformed.t1.data[low_indices] == 0.).all()
        assert (transformed.t1.data[high_indices] == 1.).all()

    def test_masking_using_label(self):
        transform = RescaleIntensity(
            out_min_max=(0., 1.), percentiles=(5, 95), masking_method='label')
        transformed = transform(self.sample)
        mask = self.sample.label.data > 0
        low_quantile = np.percentile(self.sample.t1.data[mask], 5)
        high_quantile = np.percentile(self.sample.t1.data[mask], 95)
        low_indices = (self.sample.t1.data < low_quantile).nonzero(
            as_tuple=True)
        high_indices = (self.sample.t1.data > high_quantile).nonzero(
            as_tuple=True)
        self.assertEqual(transformed.t1.data.min(), 0.)
        self.assertEqual(transformed.t1.data.max(), 1.)
        assert (transformed.t1.data[low_indices] == 0.).all()
        assert (transformed.t1.data[high_indices] == 1.).all()

    def test_out_min_higher_than_out_max(self):
        with self.assertRaises(ValueError):
            RescaleIntensity(out_min_max=(1., 0.))

    def test_too_many_values_for_out_min_max(self):
        with self.assertRaises(ValueError):
            RescaleIntensity(out_min_max=(1., 2., 3.))

    def test_wrong_out_min_max_type(self):
        with self.assertRaises(ValueError):
            RescaleIntensity(out_min_max='wrong')

    def test_min_percentile_higher_than_max_percentile(self):
        with self.assertRaises(ValueError):
            RescaleIntensity(out_min_max=(0., 1.), percentiles=(1., 0.))

    def test_too_many_values_for_percentiles(self):
        with self.assertRaises(ValueError):
            RescaleIntensity(out_min_max=(0., 1.), percentiles=(1., 2., 3.))

    def test_wrong_percentiles_type(self):
        with self.assertRaises(ValueError):
            RescaleIntensity(out_min_max=(0., 1.), percentiles='wrong')
