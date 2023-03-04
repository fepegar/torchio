import torchio as tio

from ...utils import TorchioTestCase


class TestRemoveLabels(TorchioTestCase):
    """Tests for `RemoveLabels`."""

    def test_remove(self):
        original_labels = (1, 2, 3, 4, 5, 6, 7)
        labels_to_remove = (1, 2, 5, 6)
        remaining_labels = (3, 4, 7)

        remove_labels = tio.RemoveLabels(labels_to_remove)

        tensor = TorchioTestCase.get_tensor_with_labels(original_labels)
        subject = tio.Subject(label=tio.LabelMap(tensor=tensor))
        transformed = remove_labels(subject)

        for removed_label in labels_to_remove:
            original_mask = subject.label.data == removed_label
            new_values = transformed.label.data[original_mask]
            self.assert_tensor_all_zeros(new_values)

        for remaining_label in remaining_labels:
            original_mask = subject.label.data == remaining_label
            original_values = subject.label.data[original_mask]
            output_values = transformed.label.data[original_mask]
            self.assert_tensor_equal(original_values, output_values)
