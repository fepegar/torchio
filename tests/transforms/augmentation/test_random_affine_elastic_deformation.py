import pytest
import torch
import torchio as tio

from ...utils import TorchioTestCase


class TestRandomCombinedAffineElasticDeformation(TorchioTestCase):
    """Tests for `RandomCombinedAffineElasticDeformation`."""

    def setUp(self):
        # Set image origin far from center
        super().setUp()
        affine = self.sample_subject.t1.affine
        affine[:3, 3] = 1e5

    def test_inputs_pta_gt_one(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(p=1.5)

    def test_inputs_pta_lt_zero(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(p=-1)

    def test_inputs_interpolation_int(self):
        with pytest.raises(TypeError):
            tio.RandomCombinedAffineElasticDeformation(image_interpolation=1)

    def test_inputs_interpolation(self):
        with pytest.raises(TypeError):
            tio.RandomCombinedAffineElasticDeformation(image_interpolation=0)

    def test_num_control_points_noint(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                elastic_kwargs={'num_control_points': 2.5}
            )

    def test_num_control_points_small(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                elastic_kwargs={'num_control_points': 3}
            )

    def test_max_displacement_no_num(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                elastic_kwargs={'max_displacement': None}
            )

    def test_max_displacement_negative(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                elastic_kwargs={'max_displacement': -1}
            )

    def test_wrong_locked_borders(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                elastic_kwargs={'locked_borders': -1}
            )

    def test_coarse_grid_removed(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                elastic_kwargs={'num_control_points': (4, 5, 6), 'locked_borders': 2}
            )

    def test_folding(self):
        # Assume shape is (10, 20, 30) and spacing is (1, 1, 1)
        # Then grid spacing is (10/(12-2), 20/(5-2), 30/(5-2))
        # or (1, 6.7, 10), and half is (0.5, 3.3, 5)
        transform = tio.RandomCombinedAffineElasticDeformation(
            elastic_kwargs={'num_control_points': (12, 5, 5), 'max_displacement': 6}
        )
        with pytest.warns(RuntimeWarning):
            transform(self.sample_subject)

    def test_num_control_points(self):
        tio.RandomCombinedAffineElasticDeformation(
            elastic_kwargs={'num_control_points': 5}
        )
        tio.RandomCombinedAffineElasticDeformation(
            elastic_kwargs={'num_control_points': (5, 6, 7)}
        )

    def test_max_displacement(self):
        tio.RandomCombinedAffineElasticDeformation(
            elastic_kwargs={'max_displacement': 5}
        )
        tio.RandomCombinedAffineElasticDeformation(
            elastic_kwargs={'max_displacement': (5, 6, 7)}
        )

    def test_no_displacement(self):
        transform = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={'scales': 0, 'degrees': 0, 'translation': 0},
            elastic_kwargs={'max_displacement': 0},
        )
        transformed = transform(self.sample_subject)
        self.assert_tensor_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )
        self.assert_tensor_equal(
            self.sample_subject.label.data,
            transformed.label.data,
        )

    def test_rotation_image(self):
        # Rotation around image center
        transform = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={
                'degrees': (90, 90),
                'default_pad_value': 0,
                'center': 'image',
            }
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        self.assertNotEqual(total, 0)

    def test_rotation_origin(self):
        # Rotation around far away point, image should be empty
        transform = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={
                'degrees': (90, 90),
                'default_pad_value': 0,
                'center': 'origin',
            }
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        assert total == 0

    def test_no_rotation(self):
        transform = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={
                'scales': (1, 1),
                'degrees': (0, 0),
                'default_pad_value': 0,
                'center': 'image',
            },
            elastic_kwargs={'max_displacement': 0},
        )
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )
        transform = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={
                'scales': (1, 1),
                'degrees': (180, 180),
                'default_pad_value': 0,
                'center': 'image',
            },
            elastic_kwargs={'max_displacement': 0},
        )
        transformed = transform(self.sample_subject)
        transformed = transform(transformed)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_isotropic(self):
        tio.RandomCombinedAffineElasticDeformation(affine_kwargs={'isotropic': True})(
            self.sample_subject
        )

    def test_mean(self):
        tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={'default_pad_value': 'mean'}
        )(self.sample_subject)

    def test_otsu(self):
        tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={'default_pad_value': 'otsu'}
        )(self.sample_subject)

    def test_bad_center(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(affine_kwargs={'center': 'bad'})

    def test_negative_scales(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': (-1, 1)}
            )

    def test_scale_too_large(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(affine_kwargs={'scales': 1.5})

    def test_scales_range_with_negative_min(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': (-1, 4)}
            )

    def test_wrong_scales_type(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': 'wrong'}
            )

    def test_wrong_degrees_type(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'degrees': 'wrong'}
            )

    def test_too_many_translation_values(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'translation': (-10, 4, 42)}
            )

    def test_wrong_translation_type(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'translation': 'wrong'}
            )

    def test_wrong_center(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(affine_kwargs={'center': 0})

    def test_wrong_default_pad_value(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'default_pad_value': 'wrong'}
            )

    def test_wrong_image_interpolation_type(self):
        with pytest.raises(TypeError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'image_interpolation': 0}
            )

    def test_wrong_image_interpolation_value(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'image_interpolation': 'wrong'}
            )

    def test_incompatible_args_isotropic(self):
        with pytest.raises(ValueError):
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': (0.8, 0.5, 0.1), 'isotropic': True}
            )

    def test_parse_scales(self):
        def do_assert(transform):
            assert transform.random_affine.scales == 3 * (0.9, 1.1)

        do_assert(
            tio.RandomCombinedAffineElasticDeformation(affine_kwargs={'scales': 0.1})
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': (0.9, 1.1)}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': 3 * (0.1,)}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'scales': 3 * [0.9, 1.1]}
            )
        )

    def test_parse_degrees(self):
        def do_assert(transform):
            assert transform.random_affine.degrees == 3 * (-10, 10)

        do_assert(
            tio.RandomCombinedAffineElasticDeformation(affine_kwargs={'degrees': 10})
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'degrees': (-10, 10)}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'degrees': 3 * (10,)}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'degrees': 3 * [-10, 10]}
            )
        )

    def test_parse_translation(self):
        def do_assert(transform):
            assert transform.random_affine.translation == 3 * (-10, 10)

        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'translation': 10}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'translation': (-10, 10)}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'translation': 3 * (10,)}
            )
        )
        do_assert(
            tio.RandomCombinedAffineElasticDeformation(
                affine_kwargs={'translation': 3 * [-10, 10]}
            )
        )

    def test_default_value_label_map(self):
        # From https://github.com/fepegar/torchio/issues/626
        a = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1, 3, 3, 1)
        image = tio.LabelMap(tensor=a)
        aff = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={'translation': (0, 1, 1), 'default_pad_value': 'otsu'}
        )
        transformed = aff(image)
        assert all(n in (0, 1) for n in transformed.data.flatten())

    def test_no_inverse(self):
        tensor = torch.zeros((1, 2, 2, 2))
        tensor[0, 1, 1, 1] = 1  # most RAS voxel
        expected = torch.zeros((1, 2, 2, 2))
        expected[0, 0, 1, 1] = 1
        scales = 1, 1, 1
        degrees = 0, 0, 90  # anterior should go left
        translation = 0, 0, 0
        apply_affine = tio.Affine(
            scales,
            degrees,
            translation,
        )
        transformed = apply_affine(tensor)
        self.assert_tensor_almost_equal(transformed, expected)

    def test_different_spaces(self):
        t1 = self.sample_subject.t1
        label = tio.Resample(2)(self.sample_subject.label)
        new_subject = tio.Subject(t1=t1, label=label)
        with pytest.raises(RuntimeError):
            tio.RandomCombinedAffineElasticDeformation()(new_subject)
        tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={'check_shape': False}
        )(new_subject)

    def test_transform_order(self):
        src_transform = tio.RandomCombinedAffineElasticDeformation(
            affine_kwargs={'scales': 0, 'degrees': 0, 'translation': 1},
            elastic_kwargs={'num_control_points': 5, 'max_displacement': 1},
        )

        (scales, degrees, translation), control_points = src_transform.get_params()

        max_displacement = src_transform.random_elastic.max_displacement

        transform1 = tio.CombinedAffineElasticDeformation(
            affine_first=True,
            affine_params={
                'scales': scales,
                'degrees': degrees,
                'translation': translation,
            },
            elastic_params={
                'control_points': control_points,
                'max_displacement': max_displacement,
            },
        )
        transform2 = tio.CombinedAffineElasticDeformation(
            affine_first=False,
            affine_params={
                'scales': scales,
                'degrees': degrees,
                'translation': translation,
            },
            elastic_params={
                'control_points': control_points,
                'max_displacement': max_displacement,
            },
        )

        transformed1 = transform1(self.sample_subject)
        transformed2 = transform2(self.sample_subject)
        self.assert_tensor_not_equal(transformed1.t1.data, transformed2.t1.data)
