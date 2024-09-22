import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomElasticDeformation(TorchioTestCase):
    """Tests for `RandomElasticDeformation`."""

    def test_inputs_pta_gt_one(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(p=1.5)

    def test_inputs_pta_lt_zero(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(p=-1)

    def test_inputs_interpolation_int(self):
        with pytest.raises(TypeError):
            tio.RandomElasticDeformation(image_interpolation=1)

    def test_inputs_interpolation(self):
        with pytest.raises(TypeError):
            tio.RandomElasticDeformation(image_interpolation=0)

    def test_num_control_points_noint(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(num_control_points=2.5)

    def test_num_control_points_small(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(num_control_points=3)

    def test_max_displacement_no_num(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(max_displacement=None)

    def test_max_displacement_negative(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(max_displacement=-1)

    def test_wrong_locked_borders(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(locked_borders=-1)

    def test_coarse_grid_removed(self):
        with pytest.raises(ValueError):
            tio.RandomElasticDeformation(
                num_control_points=(4, 5, 6),
                locked_borders=2,
            )

    def test_folding(self):
        # Assume shape is (10, 20, 30) and spacing is (1, 1, 1)
        # Then grid spacing is (10/(12-2), 20/(5-2), 30/(5-2))
        # or (1, 6.7, 10), and half is (0.5, 3.3, 5)
        transform = tio.RandomElasticDeformation(
            num_control_points=(12, 5, 5),
            max_displacement=6,
        )
        with pytest.warns(RuntimeWarning):
            transform(self.sample_subject)

    def test_num_control_points(self):
        tio.RandomElasticDeformation(num_control_points=5)
        tio.RandomElasticDeformation(num_control_points=(5, 6, 7))

    def test_max_displacement(self):
        tio.RandomElasticDeformation(max_displacement=5)
        tio.RandomElasticDeformation(max_displacement=(5, 6, 7))

    def test_no_displacement(self):
        transform = tio.RandomElasticDeformation(max_displacement=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )
        self.assert_tensor_equal(
            self.sample_subject.label.data,
            transformed.label.data,
        )
