from torchio.transforms import SequentialLabels
from ...utils import TorchioTestCase


class TestSequentialLabels(TorchioTestCase):
    """Tests for `SequentialLabels`."""
    def test_sequential(self):
        initial_labels = (2, 8, 9, 10, 15, 20, 100)
        transformed_labels = (1, 2, 3, 4, 5, 6, 7)

        sequential_labels = SequentialLabels()

        subject = self.get_subject_with_labels(labels=initial_labels)
        transformed = sequential_labels(subject)
        inverse_transformed = transformed.apply_inverse_transform()

        self.assertEqual(
            self.get_unique_labels(subject.label),
            set(initial_labels),
        )
        self.assertEqual(
            self.get_unique_labels(transformed.label),
            set(transformed_labels),
        )
        self.assertEqual(
            self.get_unique_labels(inverse_transformed.label),
            set(initial_labels),
        )
