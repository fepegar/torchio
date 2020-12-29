from torchio.transforms import RemoveLabels
from ...utils import TorchioTestCase


class TestRemoveLabels(TorchioTestCase):
    """Tests for `RemoveLabels`."""
    def test_remove(self):
        initial_labels = (1, 2, 3, 4, 5, 6, 7)
        labels_to_remove = (1, 2, 5, 6)
        remaining_labels = (3, 4, 7)

        remove_labels = RemoveLabels(labels_to_remove)

        subject = self.get_subject_with_labels(labels=initial_labels)
        transformed = remove_labels(subject)
        inverse_transformed = transformed.apply_inverse_transform(warn=False)
        self.assertEqual(
            self.get_unique_labels(subject.label),
            set(initial_labels),
        )
        self.assertEqual(
            self.get_unique_labels(transformed.label),
            set(remaining_labels),
        )
        self.assertEqual(
            self.get_unique_labels(inverse_transformed.label),
            set(remaining_labels),
        )
