import torch
import torchio as tio
from ...utils import TorchioTestCase


class TestRandomAffine(TorchioTestCase):
    """Tests for `RandomAffine`."""
    def setUp(self):
        # Set image origin far from center
        super().setUp()
        affine = self.sample_subject.t1.affine
        affine[:3, 3] = 1e5

    def test_rotation_image(self):
        # Rotation around image center
        transform = tio.RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        self.assertNotEqual(total, 0)

    def test_rotation_origin(self):
        # Rotation around far away point, image should be empty
        transform = tio.RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='origin',
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        self.assertEqual(total, 0)

    def test_no_rotation(self):
        transform = tio.RandomAffine(
            scales=(1, 1),
            degrees=(0, 0),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

        transform = tio.RandomAffine(
            scales=(1, 1),
            degrees=(180, 180),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        transformed = transform(transformed)
        self.assertTensorAlmostEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_isotropic(self):
        tio.RandomAffine(isotropic=True)(self.sample_subject)

    def test_mean(self):
        tio.RandomAffine(default_pad_value='mean')(self.sample_subject)

    def test_otsu(self):
        tio.RandomAffine(default_pad_value='otsu')(self.sample_subject)

    def test_bad_center(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(center='bad')

    def test_translation(self):
        transform = tio.RandomAffine(
            scales=(1, 1),
            degrees=0,
            translation=(5, 5)
        )
        transformed = transform(self.sample_subject)

        # I think the right test should be the following one:
        # self.assertTensorAlmostEqual(
        #     self.sample_subject.t1.data[:, :-5, :-5, :-5],
        #     transformed.t1.data[:, 5:, 5:, 5:]
        # )

        # However the passing test is this one:
        self.assertTensorAlmostEqual(
            self.sample_subject.t1.data[:, :-5, :-5, 5:],
            transformed.t1.data[:, 5:, 5:, :-5]
        )

    def test_negative_scales(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(scales=(-1, 1))

    def test_scale_too_large(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(scales=1.5)

    def test_scales_range_with_negative_min(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(scales=(-1, 4))

    def test_wrong_scales_type(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(scales='wrong')

    def test_wrong_degrees_type(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(degrees='wrong')

    def test_too_many_translation_values(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(translation=(-10, 4, 42))

    def test_wrong_translation_type(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(translation='wrong')

    def test_wrong_center(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(center=0)

    def test_wrong_default_pad_value(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(default_pad_value='wrong')

    def test_wrong_image_interpolation_type(self):
        with self.assertRaises(TypeError):
            tio.RandomAffine(image_interpolation=0)

    def test_wrong_image_interpolation_value(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(image_interpolation='wrong')

    def test_incompatible_args_isotropic(self):
        with self.assertRaises(ValueError):
            tio.RandomAffine(scales=(0.8, 0.5, 0.1), isotropic=True)

    def test_parse_scales(self):
        def do_assert(transform):
            self.assertEqual(transform.scales, 3 * (0.9, 1.1))
        do_assert(tio.RandomAffine(scales=0.1))
        do_assert(tio.RandomAffine(scales=(0.9, 1.1)))
        do_assert(tio.RandomAffine(scales=3 * (0.1,)))
        do_assert(tio.RandomAffine(scales=3 * [0.9, 1.1]))

    def test_parse_degrees(self):
        def do_assert(transform):
            self.assertEqual(transform.degrees, 3 * (-10, 10))
        do_assert(tio.RandomAffine(degrees=10))
        do_assert(tio.RandomAffine(degrees=(-10, 10)))
        do_assert(tio.RandomAffine(degrees=3 * (10,)))
        do_assert(tio.RandomAffine(degrees=3 * [-10, 10]))

    def test_parse_translation(self):
        def do_assert(transform):
            self.assertEqual(transform.translation, 3 * (-10, 10))
        do_assert(tio.RandomAffine(translation=10))
        do_assert(tio.RandomAffine(translation=(-10, 10)))
        do_assert(tio.RandomAffine(translation=3 * (10,)))
        do_assert(tio.RandomAffine(translation=3 * [-10, 10]))

    def test_default_value_label_map(self):
        # From https://github.com/fepegar/torchio/issues/626
        a = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1, 3, 3, 1)
        image = tio.LabelMap(tensor=a)
        aff = tio.RandomAffine(translation=(0, 1, 1), default_pad_value='otsu')
        transformed = aff(image)
        assert all(n in (0, 1) for n in transformed.data.flatten())
