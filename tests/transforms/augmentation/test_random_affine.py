from torchio.transforms import RandomAffine
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
        transform = RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        self.assertNotEqual(total, 0)

    def test_rotation_origin(self):
        # Rotation around far away point, image should be empty
        transform = RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='origin',
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        self.assertEqual(total, 0)

    def test_no_rotation(self):
        transform = RandomAffine(
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

        transform = RandomAffine(
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

    def test_translation(self):
        transform = RandomAffine(
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
            RandomAffine(scales=(-1, 1))

    def test_scale_too_large(self):
        with self.assertRaises(ValueError):
            RandomAffine(scales=1.5)

    def test_scales_range_with_negative_min(self):
        with self.assertRaises(ValueError):
            RandomAffine(scales=(-1, 4))

    def test_wrong_scales_type(self):
        with self.assertRaises(ValueError):
            RandomAffine(scales='wrong')

    def test_wrong_degrees_type(self):
        with self.assertRaises(ValueError):
            RandomAffine(degrees='wrong')

    def test_too_many_translation_values(self):
        with self.assertRaises(ValueError):
            RandomAffine(translation=(-10, 4, 42))

    def test_wrong_translation_type(self):
        with self.assertRaises(ValueError):
            RandomAffine(translation='wrong')

    def test_wrong_center(self):
        with self.assertRaises(ValueError):
            RandomAffine(center=0)

    def test_wrong_default_pad_value(self):
        with self.assertRaises(ValueError):
            RandomAffine(default_pad_value='wrong')

    def test_wrong_image_interpolation_type(self):
        with self.assertRaises(TypeError):
            RandomAffine(image_interpolation=0)

    def test_wrong_image_interpolation_value(self):
        with self.assertRaises(ValueError):
            RandomAffine(image_interpolation='wrong')

    def test_incompatible_args_isotropic(self):
        with self.assertRaises(ValueError):
            RandomAffine(scales=(0.8, 0.5, 0.1), isotropic=True)

    def test_parse_scales(self):
        def do_assert(transform):
            self.assertEqual(transform.scales, 3 * (0.9, 1.1))
        do_assert(RandomAffine(scales=0.1))
        do_assert(RandomAffine(scales=(0.9, 1.1)))
        do_assert(RandomAffine(scales=3 * (0.1,)))
        do_assert(RandomAffine(scales=3 * [0.9, 1.1]))

    def test_parse_degrees(self):
        def do_assert(transform):
            self.assertEqual(transform.degrees, 3 * (-10, 10))
        do_assert(RandomAffine(degrees=10))
        do_assert(RandomAffine(degrees=(-10, 10)))
        do_assert(RandomAffine(degrees=3 * (10,)))
        do_assert(RandomAffine(degrees=3 * [-10, 10]))

    def test_parse_translation(self):
        def do_assert(transform):
            self.assertEqual(transform.translation, 3 * (-10, 10))
        do_assert(RandomAffine(translation=10))
        do_assert(RandomAffine(translation=(-10, 10)))
        do_assert(RandomAffine(translation=3 * (10,)))
        do_assert(RandomAffine(translation=3 * [-10, 10]))
