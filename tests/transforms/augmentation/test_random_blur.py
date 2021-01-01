from torchio import RandomBlur
from ...utils import TorchioTestCase


class TestRandomBlur(TorchioTestCase):
    """Tests for `RandomBlur`."""
    def test_no_blurring(self):
        transform = RandomBlur(std=0)
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_blurring(self):
        transform = RandomBlur(std=(1, 3))
        transformed = transform(self.sample_subject)
        self.assertTensorNotEqual(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_negative_std(self):
        with self.assertRaises(ValueError):
            RandomBlur(std=-2)

    def test_std_range_with_negative_min(self):
        with self.assertRaises(ValueError):
            RandomBlur(std=(-0.5, 4))

    def test_wrong_std_type(self):
        with self.assertRaises(ValueError):
            RandomBlur(std='wrong')

    def test_parse_stds(self):
        def do_assert(transform):
            self.assertEqual(transform.std_ranges, 3 * (0, 1))
        do_assert(RandomBlur(std=1))
        do_assert(RandomBlur(std=(0, 1)))
        do_assert(RandomBlur(std=3 * (1,)))
        do_assert(RandomBlur(std=3 * [0, 1]))
