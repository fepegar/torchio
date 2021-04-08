import torch
import torchio as tio
import numpy as np
from ...utils import TorchioTestCase


class TestRescaleIntensity(TorchioTestCase):
    """Tests for :class:`tio.RescaleIntensity` class."""

    def test_rescale_to_same_intentisy(self):
        min_t1 = float(self.sample_subject.t1.data.min())
        max_t1 = float(self.sample_subject.t1.data.max())
        transform = tio.RescaleIntensity(out_min_max=(min_t1, max_t1))
        transformed = transform(self.sample_subject)
        assert np.allclose(
            transformed.t1.data,
            self.sample_subject.t1.data,
            rtol=0,
            atol=1e-05,
        )

    def test_min_max(self):
        transform = tio.RescaleIntensity(out_min_max=(0, 1))
        transformed = transform(self.sample_subject)
        self.assertEqual(transformed.t1.data.min(), 0)
        self.assertEqual(transformed.t1.data.max(), 1)

    def test_percentiles(self):
        low_quantile = np.percentile(self.sample_subject.t1.data, 5)
        high_quantile = np.percentile(self.sample_subject.t1.data, 95)
        low_indices = (self.sample_subject.t1.data < low_quantile).nonzero(
            as_tuple=True)
        high_indices = (self.sample_subject.t1.data > high_quantile).nonzero(
            as_tuple=True)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(5, 95))
        transformed = rescale(self.sample_subject)
        assert (transformed.t1.data[low_indices] == 0).all()
        assert (transformed.t1.data[high_indices] == 1).all()

    def test_masking_using_label(self):
        transform = tio.RescaleIntensity(
            out_min_max=(0, 1), percentiles=(5, 95), masking_method='label')
        transformed = transform(self.sample_subject)
        mask = self.sample_subject.label.data > 0
        low_quantile = np.percentile(self.sample_subject.t1.data[mask], 5)
        high_quantile = np.percentile(self.sample_subject.t1.data[mask], 95)
        low_indices = (self.sample_subject.t1.data < low_quantile).nonzero(
            as_tuple=True)
        high_indices = (self.sample_subject.t1.data > high_quantile).nonzero(
            as_tuple=True)
        self.assertEqual(transformed.t1.data.min(), 0)
        self.assertEqual(transformed.t1.data.max(), 1)
        assert (transformed.t1.data[low_indices] == 0).all()
        assert (transformed.t1.data[high_indices] == 1).all()

    def test_ct(self):
        ct_max = 1500
        ct_min = -2000
        ct_range = ct_max - ct_min
        tensor = torch.rand(1, 30, 30, 30) * ct_range + ct_min
        ct = tio.ScalarImage(tensor=tensor)
        ct_air = -1000
        ct_bone = 1000
        rescale = tio.RescaleIntensity(
            out_min_max=(-1, 1),
            in_min_max=(ct_air, ct_bone),
        )
        rescaled = rescale(ct)
        assert rescaled.data.min() < -1
        assert rescaled.data.max() > 1

    def test_out_min_higher_than_out_max(self):
        with self.assertRaises(ValueError):
            tio.RescaleIntensity(out_min_max=(1, 0))

    def test_too_many_values_for_out_min_max(self):
        with self.assertRaises(ValueError):
            tio.RescaleIntensity(out_min_max=(1, 2, 3))

    def test_wrong_out_min_max_type(self):
        with self.assertRaises(ValueError):
            tio.RescaleIntensity(out_min_max='wrong')

    def test_min_percentile_higher_than_max_percentile(self):
        with self.assertRaises(ValueError):
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 0))

    def test_too_many_values_for_percentiles(self):
        with self.assertRaises(ValueError):
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 2, 3))

    def test_wrong_percentiles_type(self):
        with self.assertRaises(ValueError):
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles='wrong')
