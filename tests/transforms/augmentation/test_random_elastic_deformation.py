import torchio
from torchio.transforms import RandomElasticDeformation
from ...utils import TorchioTestCase


class TestRandomElasticDeformation(TorchioTestCase):
    """Tests for `RandomElasticDeformation`."""

    def test_random_elastic_deformation(self):
        transform = RandomElasticDeformation(
            seed=42,
        )
        keys = ('t1', 't2', 'label')
        fixtures = 2794.82470703125, 2763.881591796875, 2751
        transformed = transform(self.sample)
        for key, fixture in zip(keys, fixtures):
            data = transformed[key][torchio.DATA]
            total = data.sum().item()
            self.assertAlmostEqual(total, fixture)

    def test_inputs_pta_gt_one(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(proportion_to_augment=1.5)

    def test_inputs_pta_lt_zero(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(proportion_to_augment=-1)

    def test_inputs_interpolation_int(self):
        with self.assertRaises(TypeError):
            RandomElasticDeformation(image_interpolation=1)

    def test_inputs_interpolation_string(self):
        with self.assertRaises(TypeError):
            RandomElasticDeformation(image_interpolation='linear')

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            RandomElasticDeformation(deformation_std=15)

    def test_num_control_points_noint(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(num_control_points=2.5)

    def test_num_control_points_small(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(num_control_points=3)

    def test_max_displacement_no_num(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(max_displacement=None)

    def test_max_displacement_negative(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(max_displacement=-1)

    def test_wrong_locked_borders(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(locked_borders=-1)

    def test_coarse_grid_removed(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(
                num_control_points=(4, 5, 6),
                locked_borders=2,
            )

    def test_folding(self):
        # Assume shape is (10, 20, 30)
        transform = RandomElasticDeformation(
            num_control_points=(12, 5, 5),
            max_displacement=6,
        )
        with self.assertWarns(UserWarning):
            transformed = transform(self.sample)
