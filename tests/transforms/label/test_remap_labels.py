from torchio.transforms import RemapLabels
from ...utils import TorchioTestCase


class TestRemapLabels(TorchioTestCase):
    """Tests for `RemapLabels`."""
    def test_remap(self):
        remapping = {1: 2, 2: 1, 5: 10, 6: 11}
        remap_labels = RemapLabels(remapping=remapping)

        subject = self.get_subject_with_labels(labels=remapping.keys())
        transformed = remap_labels(subject)
        inverse_transformed = transformed.apply_inverse_transform()

        self.assertEqual(
            self.get_unique_labels(subject.label),
            set(remapping.keys()),
        )
        self.assertEqual(
            self.get_unique_labels(transformed.label),
            set(remapping.values()),
        )
        self.assertEqual(
            self.get_unique_labels(inverse_transformed.label),
            set(remapping.keys()),
        )
