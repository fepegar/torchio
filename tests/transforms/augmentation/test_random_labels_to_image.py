from torchio.transforms import RandomLabelsToImage
from ...utils import TorchioTestCase


class TestRandomLabelsToImage(TorchioTestCase):
    """Tests for `RandomLabelsToImage`."""
    def test_random_simulation(self):
        """The transform runs without error and an 'image_from_labels' key is
        present in the transformed subject."""
        transform = RandomLabelsToImage(label_key='label')
        transformed = transform(self.sample_subject)
        self.assertIn('image_from_labels', transformed)

    def test_deterministic_simulation(self):
        """The transform creates an image where values are equal to given
        mean if standard deviation is zero.
        Using a label map."""
        transform = RandomLabelsToImage(
            label_key='label',
            mean=[0.5, 2],
            std=[0, 0]
        )
        transformed = transform(self.sample_subject)
        self.assertTensorEqual(
            transformed['image_from_labels'].data == 0.5,
            self.sample_subject['label'].data == 0
        )
        self.assertTensorEqual(
            transformed['image_from_labels'].data == 2,
            self.sample_subject['label'].data == 1
        )

    def test_deterministic_simulation_with_discretized_label_map(self):
        """The transform creates an image where values are equal to given mean
        if standard deviation is zero.
        Using a discretized label map."""
        transform = RandomLabelsToImage(
            label_key='label',
            mean=[0.5, 2],
            std=[0, 0],
            discretize=True
        )
        transformed = transform(self.sample_subject)
        self.assertTensorEqual(
            transformed['image_from_labels'].data == 0.5,
            self.sample_subject['label'].data == 0
        )
        self.assertTensorEqual(
            transformed['image_from_labels'].data == 2,
            self.sample_subject['label'].data == 1
        )

    def test_deterministic_simulation_with_pv_map(self):
        """The transform creates an image where values are equal to given
        mean weighted by partial-volume if standard deviation is zero."""
        subject = self.get_subject_with_partial_volume_label_map(components=2)
        transform = RandomLabelsToImage(
            label_key='label',
            mean=[0.5, 1],
            std=[0, 0]
        )
        transformed = transform(subject)
        self.assertTensorAlmostEqual(
            transformed['image_from_labels'].data[0],
            subject['label'].data[0] * 0.5 + subject['label'].data[1] * 1
        )
        self.assertEqual(
            transformed['image_from_labels'].data.shape,
            (1, 10, 20, 30)
        )

    def test_deterministic_simulation_with_discretized_pv_map(self):
        """The transform creates an image where values are equal to given mean
        if standard deviation is zero.
        Using a discretized partial-volume label map."""
        subject = self.get_subject_with_partial_volume_label_map()
        transform = RandomLabelsToImage(
            label_key='label',
            mean=[0.5],
            std=[0],
            discretize=True
        )
        transformed = transform(subject)
        self.assertTensorAlmostEqual(
            transformed['image_from_labels'].data,
            (subject['label'].data > 0) * 0.5
        )

    def test_filling(self):
        """The transform can fill in the generated image with an already
        existing image.
        Using a label map."""
        transform = RandomLabelsToImage(
            label_key='label',
            image_key='t1',
            used_labels=[1]
        )
        t1_indices = self.sample_subject['label'].data == 0
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(
            transformed['t1'].data[t1_indices],
            self.sample_subject['t1'].data[t1_indices]
        )

    def test_filling_with_discretized_label_map(self):
        """The transform can fill in the generated image with an already
        existing image.
        Using a discretized label map."""
        transform = RandomLabelsToImage(
            label_key='label',
            image_key='t1',
            discretize=True,
            used_labels=[1]
        )
        t1_indices = self.sample_subject['label'].data < 0.5
        transformed = transform(self.sample_subject)
        self.assertTensorAlmostEqual(
            transformed['t1'].data[t1_indices],
            self.sample_subject['t1'].data[t1_indices]
        )

    def test_filling_with_discretized_pv_label_map(self):
        """The transform can fill in the generated image with an already
        existing image.
        Using a discretized partial-volume label map."""
        subject = self.get_subject_with_partial_volume_label_map(components=2)
        transform = RandomLabelsToImage(
            label_key='label',
            image_key='t1',
            discretize=True,
            used_labels=[1]
        )
        t1_indices = subject['label'].data.argmax(dim=0) == 0
        transformed = transform(subject)
        self.assertTensorAlmostEqual(
            transformed['t1'].data[0][t1_indices],
            subject['t1'].data[0][t1_indices]
        )

    def test_filling_without_any_hole(self):
        """The transform does not fill anything if there is no hole."""
        transform = RandomLabelsToImage(
            label_key='label',
            image_key='t1',
            default_std=0.,
            default_mean=-1.
        )
        original_t1 = self.sample_subject.t1.data.clone()
        transformed = transform(self.sample_subject)
        self.assertTensorNotEqual(original_t1, transformed.t1.data)

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

    def test_with_wrong_used_labels_type(self):
        """The transform raises an error if a wrong type is given for
        used_labels."""
        with self.assertRaises(TypeError):
            RandomLabelsToImage(label_key='label', used_labels=42)

    def test_with_wrong_used_labels_elements_type(self):
        """The transform raises an error if wrong type are given for
        used_labels elements."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', used_labels=['wrong'])

    def test_with_wrong_mean_type(self):
        """The transform raises an error if wrong type is given for mean."""
        with self.assertRaises(TypeError):
            RandomLabelsToImage(label_key='label', mean=42)

    def test_with_wrong_mean_elements_type(self):
        """The transform raises an error if wrong type are given for
        mean elements."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', mean=['wrong'])

    def test_with_wrong_std_type(self):
        """The transform raises an error if wrong type is given for std."""
        with self.assertRaises(TypeError):
            RandomLabelsToImage(label_key='label', std=42)

    def test_with_wrong_std_elements_type(self):
        """The transform raises an error if wrong type are given for
        std elements."""
        with self.assertRaises(ValueError):
            RandomLabelsToImage(label_key='label', std=['wrong'])

    def test_mean_and_std_len_not_matching(self):
        """The transform raises an error if mean and std length don't match."""
        with self.assertRaises(AssertionError):
            RandomLabelsToImage(label_key='label', mean=[0], std=[0, 1])

    def test_mean_and_used_labels_len_not_matching(self):
        """The transform raises an error if mean and used_labels
         length don't match."""
        with self.assertRaises(AssertionError):
            RandomLabelsToImage(
                label_key='label',
                mean=[0],
                used_labels=[0, 1],
            )

    def test_std_and_used_labels_len_not_matching(self):
        """The transform raises an error if std and used_labels
         length don't match."""
        with self.assertRaises(AssertionError):
            RandomLabelsToImage(label_key='label', std=[0], used_labels=[0, 1])

    def test_mean_not_matching_number_of_labels(self):
        """The transform raises an error at runtime if mean length
        does not match label numbers."""
        transform = RandomLabelsToImage(label_key='label', mean=[0])
        with self.assertRaises(RuntimeError):
            transform(self.sample_subject)

    def test_std_not_matching_number_of_labels(self):
        """The transform raises an error at runtime if std length
        does not match label numbers."""
        transform = RandomLabelsToImage(label_key='label', std=[1, 2, 3])
        with self.assertRaises(RuntimeError):
            transform(self.sample_subject)
