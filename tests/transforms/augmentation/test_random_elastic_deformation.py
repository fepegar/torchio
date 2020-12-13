from torchio.transforms import RandomElasticDeformation
from ...utils import TorchioTestCase


class TestRandomElasticDeformation(TorchioTestCase):
    """Tests for `RandomElasticDeformation`."""

    def test_inputs_pta_gt_one(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(p=1.5)

    def test_inputs_pta_lt_zero(self):
        with self.assertRaises(ValueError):
            RandomElasticDeformation(p=-1)

    def test_inputs_interpolation_int(self):
        with self.assertRaises(TypeError):
            RandomElasticDeformation(image_interpolation=1)

    def test_inputs_interpolation(self):
        with self.assertRaises(TypeError):
            RandomElasticDeformation(image_interpolation=0)

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
        # Assume shape is (10, 20, 30) and spacing is (1, 1, 1)
        # Then grid spacing is (10/(12-2), 20/(5-2), 30/(5-2))
        # or (1, 6.7, 10), and half is (0.5, 3.3, 5)
        transform = RandomElasticDeformation(
            num_control_points=(12, 5, 5),
            max_displacement=6,
        )
        with self.assertWarns(RuntimeWarning):
            transform(self.sample_subject)

    def test_num_control_points(self):
        RandomElasticDeformation(num_control_points=5)
        RandomElasticDeformation(num_control_points=(5, 6, 7))

    def test_max_displacement(self):
        RandomElasticDeformation(max_displacement=5)
        RandomElasticDeformation(max_displacement=(5, 6, 7))

    def test_no_displacement(self):
        transform = RandomElasticDeformation(max_displacement=0)
        transformed = transform(self.sample_subject)
        self.assertTensorEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )
        self.assertTensorEqual(
            self.sample_subject.label.data,
            transformed.label.data,
        )
