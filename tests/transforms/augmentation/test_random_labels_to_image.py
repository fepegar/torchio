from torchio.transforms import RandomLabelsToImage
from torchio import DATA, AFFINE
from ...utils import TorchioTestCase
from numpy.testing import assert_array_equal


class TestRandomLabelsToImage(TorchioTestCase):
    """Tests for `RandomLabelsToImage`."""
    def test_random_simulation(self):
        """The transform runs without error and an 'image' key is
        present in the transformed sample."""
        transform = RandomLabelsToImage(label_key='label')
        transformed = transform(self.sample)
        self.assertIn('image', transformed)

    def test_deterministic_simulation(self):
        """The transform creates an image where values are equal to given
        mean if standard deviation is zero.
        Using a label map."""
        transform = RandomLabelsToImage(
            label_key='label',
            gaussian_parameters={1: {'mean': 0.5, 'std': 0}}
        )
        transformed = transform(self.sample)
        assert_array_equal(
            transformed['image'][DATA] == 0.5,
            self.sample['label'][DATA] == 1
        )

    def test_deterministic_simulation_with_pv_label_map(self):
        """The transform creates an image where values are equal to given mean
        if standard deviation is zero.
        Using a PV label map."""
        transform = RandomLabelsToImage(
            pv_label_keys=['label'],
            gaussian_parameters={'label': {'mean': 0.5, 'std': 0}}
        )
        transformed = transform(self.sample)
        assert_array_equal(
            transformed['image'][DATA] == 0.5,
            self.sample['label'][DATA] == 1
        )

    def test_deterministic_simulation_with_binary_pv_label_map(self):
        """The transform creates an image where values are equal to given mean
        if standard deviation is zero.
        Using a discretized PV label map."""
        transform = RandomLabelsToImage(
            pv_label_keys=['label'],
            gaussian_parameters={'label': {'mean': 0.5, 'std': 0}},
            discretize=True
        )
        transformed = transform(self.sample)
        assert_array_equal(
            transformed['image'][DATA] == 0.5,
            self.sample['label'][DATA] == 1
        )

    def test_filling(self):
        """The transform can fill in the generated image with an already
        existing image.
        Using a label map."""
        transform = RandomLabelsToImage(
            label_key='label',
            image_key='t1',
            gaussian_parameters={0: {'mean': 0.0, 'std': 0}}
        )
        t1_indices = self.sample['label'][DATA] == 0
        transformed = transform(self.sample)
        assert_array_equal(
            transformed['t1'][DATA][t1_indices],
            self.sample['t1'][DATA][t1_indices]
        )

    def test_filling_with_pv_label_map(self):
        """The transform can fill in the generated image with an already
        existing image.
        Using a PV label map."""
        transform = RandomLabelsToImage(
            pv_label_keys=['label'],
            image_key='t1'
        )
        t1_indices = self.sample['label'][DATA] == 0
        transformed = transform(self.sample)
        assert_array_equal(
            transformed['t1'][DATA][t1_indices],
            self.sample['t1'][DATA][t1_indices]
        )

    def test_filling_with_binary_pv_label_map(self):
        """The transform can fill in the generated image with an already
        existing image.
        Using a discretized PV label map."""
        transform = RandomLabelsToImage(
            pv_label_keys=['label'],
            image_key='t1',
            discretize=True
        )
        t1_indices = self.sample['label'][DATA] == 0
        transformed = transform(self.sample)
        assert_array_equal(
            transformed['t1'][DATA][t1_indices],
            self.sample['t1'][DATA][t1_indices]
        )

    def test_missing_label_key_and_pv_label_keys(self):
        """The transform raises an error if both label_key and pv_label_keys
         are None."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage()

    def test_with_both_label_key_and_pv_label_keys(self):
        """The transform raises an error if both label_key and pv_label_keys
        are set."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', pv_label_keys=['label'])

    def test_with_bad_default_mean_range(self):
        """The transform raises an error if default_mean is not a
        single value nor a tuple of two values."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', default_mean=(0, 1, 2))

    def test_with_bad_default_mean_type(self):
        """The transform raises an error if default_mean has the wrong type."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', default_mean='wrong')

    def test_with_bad_default_std_range(self):
        """The transform raises an error if default_std is not a
        single value nor a tuple of two values."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', default_std=(0, 1, 2))

    def test_with_bad_default_std_type(self):
        """The transform raises an error if default_std has the wrong type."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', default_std='wrong')

    def test_with_wrong_label_key_type(self):
        """The transform raises an error if a wrong type is given for
        label_key."""
        with self.assertRaises(TypeError):
            RandomLabelsToImage(label_key=42)

    def test_with_wrong_pv_label_keys_type(self):
        """The transform raises an error if a wrong type is given for
        pv_label_keys."""
        with self.assertRaises(TypeError):
            RandomLabelsToImage(pv_label_keys=42)

    def test_with_wrong_pv_label_keys_elements_type(self):
        """The transform raises an error if wrong type are given for
        pv_label_keys elements."""
        with self.assertRaises(TypeError):
            RandomLabelsToImage(pv_label_keys=[42, 27])

    def test_with_inconsistent_pv_label_maps_shapes(self):
        """The transform raises an error if PV label maps have
        inconsistent shapes."""
        transform = RandomLabelsToImage(
            pv_label_keys=['label', 'label2'],
        )
        sample = self.get_inconsistent_sample()
        with self.assertRaises(RuntimeError):
            transform(sample)

    def test_with_inconsistent_pv_label_maps_affines(self):
        """The transform raises a warning if PV label maps have
        inconsistent affines."""
        transform = RandomLabelsToImage(
            pv_label_keys=['label', 'label2'],
        )
        sample = self.get_inconsistent_sample()
        sample.load()  # otherwise sample['label2'] data wil be loaded later
        sample['label2'][DATA] = sample['label'][DATA].clone()
        sample['label2'][AFFINE][0, 0] = -1
        with self.assertRaises(RuntimeWarning):
            transform(sample)
